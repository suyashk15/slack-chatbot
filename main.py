from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.signature import SignatureVerifier
from slack_sdk.errors import SlackApiError
from openai import AsyncOpenAI
import sqlite3
import aiosqlite
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict
import json
import httpx
from urllib.parse import urlencode
from fastapi.responses import RedirectResponse
import logging
from logging.handlers import RotatingFileHandler
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Database setup
DB_NAME = "slack_messages.db"

def init_db():
    logger.info("Initializing database...")
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT NOT NULL UNIQUE,
                    workspace_token TEXT NOT NULL
                )
            """)
        logger.info("Database initialization successful")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    try:
        init_db()
        yield
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Application shutting down...")

app = FastAPI(lifespan=lifespan)

# Initialize clients
try:
    signature_verifier = SignatureVerifier(os.environ["SLACK_SIGNING_SECRET"])
    logger.info("Slack signature verifier initialized")
except KeyError:
    logger.error("SLACK_SIGNING_SECRET environment variable not found")
    raise

SLACK_CLIENT_ID = os.getenv("SLACK_CLIENT_ID", "placeholder-client-id")
SLACK_CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET", "placeholder-client-secret")
SLACK_OAUTH_REDIRECT_URI = os.getenv("SLACK_OAUTH_REDIRECT_URI", "http://localhost:8000/slack/oauth/callback")
SLACK_APP_ID = os.getenv("SLACK_APP_ID", "placeholder-app-id")

try:
    openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=30.0
    )
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"OpenAI client initialization failed: {e}", exc_info=True)
    raise

async def get_slack_client(workspace_id: str) -> AsyncWebClient:
    logger.debug(f"Getting Slack client for workspace: {workspace_id}")
    try:
        async with aiosqlite.connect(DB_NAME) as db:
            cursor = await db.execute(
                "SELECT workspace_token FROM workspaces WHERE workspace_id = ?",
                (workspace_id,))
            result = await cursor.fetchone()
            if result:
                logger.debug(f"Found token for workspace: {workspace_id}")
                return AsyncWebClient(token=result[0])
            else:
                logger.error(f"No token found for workspace_id: {workspace_id}")
                raise Exception(f"Workspace token not found for workspace_id: {workspace_id}")
    except Exception as e:
        logger.error(f"Error getting Slack client: {e}", exc_info=True)
        raise

async def store_message(channel: str, user: str, text: str):
    logger.debug(f"Storing message from user {user} in channel {channel}")
    try:
        async with aiosqlite.connect(DB_NAME) as db:
            await db.execute(
                """
                INSERT INTO messages (channel, user_id, message)
                VALUES (?, ?, ?)
                """,
                (channel, user, text)
            )
            await db.commit()
            logger.debug("Message stored successfully")
    except Exception as e:
        logger.error(f"Error storing message: {e}", exc_info=True)
        raise

async def get_conversation_history(channel_id: str) -> List[Dict]:
    logger.debug(f"Fetching conversation history for channel: {channel_id}")
    try:
        async with aiosqlite.connect(DB_NAME) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                """
                SELECT * FROM messages
                WHERE channel = ?
                ORDER BY timestamp DESC
                LIMIT 5
                """,
                (channel_id,)
            )
            messages = await cursor.fetchall()
            logger.debug(f"Retrieved {len(messages)} messages from history")
            return [dict(msg) for msg in reversed(messages)]
    except Exception as e:
        logger.error(f"Error fetching conversation history: {e}", exc_info=True)
        raise

# Temporary in-memory store for processed event timestamps
processed_events = set()

@app.post("/")
async def slack_events(request: Request):
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    logger.info("Received Slack event")
    logger.debug(f"Request timestamp: {timestamp}")

    try:
        if not signature_verifier.is_valid(
            body=body,
            timestamp=timestamp,
            signature=signature
        ):
            logger.warning("Invalid request signature received")
            raise HTTPException(status_code=400, detail="Invalid request signature")

        event_data = json.loads(body)
        logger.debug(f"Event type: {event_data.get('type')}")

        if event_data.get("type") == "url_verification":
            logger.info("Handling URL verification challenge")
            return {"challenge": event_data["challenge"]}

        event = event_data.get("event", {})
        event_ts = event.get("ts")

        # Ensure the event is processed only once
        if event_ts and event_ts in processed_events:
            logger.info(f"Skipping duplicate event: {event_ts}")
            return {"ok": True}

        processed_events.add(event_ts)

        if event.get("type") == "app_mention":
            channel_id = event["channel"]
            user_id = event["user"]
            text = event["text"]
            thread_ts = event.get("thread_ts", event["ts"])
            workspace_id = event_data.get("team_id")

            logger.info(f"Processing app mention from user {user_id} in channel {channel_id}")

            await store_message(channel_id, user_id, text)
            previous_messages = await get_conversation_history(channel_id)

            try:
                slack_client = await get_slack_client(workspace_id)

                # Format conversation history for the LLM
                conversation_history = "\n".join([
                    f"User: {msg['message']}" for msg in previous_messages
                ])

                logger.info("Calling OpenAI API")
                response = await openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant in a Slack channel. Keep responses concise and relevant."
                        },
                        {
                            "role": "user",
                            "content": f"""Previous conversation:{conversation_history}
                                        Current message: {text}
                                        Please provide a helpful response:"""
                        }
                    ]
                )
                logger.debug("OpenAI API response received")

                # Send response back to Slack
                logger.info(f"Sending response to channel {channel_id}")
                await slack_client.chat_postMessage(
                    channel=channel_id,
                    text=response.choices[0].message.content,
                    thread_ts=thread_ts
                )
                logger.info(f"Response sent successfully to channel: {channel_id}")

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await slack_client.chat_postMessage(
                    channel=user_id,
                    text="Sorry, I encountered an error processing your request."
                )

        return {"ok": True}
    except Exception as e:
        logger.error(f"Unhandled error in slack_events: {e}", exc_info=True)
        raise

@app.get("/slack/install")
async def slack_install():
    logger.info("Handling Slack installation request")
    params = {
        "client_id": SLACK_CLIENT_ID,
        "scope": "app_mentions:read,chat:write,channels:history,channels:join",
        "user_scope": "chat:write",
        "redirect_uri": SLACK_OAUTH_REDIRECT_URI
    }
    auth_url = f"https://slack.com/oauth/v2/authorize?{urlencode(params)}"
    logger.debug(f"Redirecting to Slack OAuth URL: {auth_url}")
    return RedirectResponse(url=auth_url, status_code=302)

@app.get("/slack/oauth/callback")
async def slack_oauth_callback(code: str):
    logger.info("Processing OAuth callback")
    try:
        async with httpx.AsyncClient() as client:
            logger.debug("Requesting access token from Slack")
            response = await client.post(
                "https://slack.com/api/oauth.v2.access",
                data={
                    "client_id": SLACK_CLIENT_ID,
                    "client_secret": SLACK_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": SLACK_OAUTH_REDIRECT_URI
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

        data = response.json()

        if not data.get("ok"):
            logger.error(f"OAuth failed: {data}")
            return {"error": "OAuth failed", "details": data}

        # Store the workspace token and other details in the database
        workspace_id = data["team"]["id"]
        workspace_token = data["access_token"]
        logger.info(f"Storing workspace token for workspace: {workspace_id}")

        async with aiosqlite.connect(DB_NAME) as db:
            await db.execute(
                """
                INSERT INTO workspaces (workspace_id, workspace_token)
                VALUES (?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET workspace_token = excluded.workspace_token
                """,
                (workspace_id, workspace_token)
            )
            await db.commit()
            logger.info("Workspace token stored successfully")

        return RedirectResponse(url=f"https://slack.com/app_redirect?app={SLACK_APP_ID}")
    except Exception as e:
        logger.error(f"Error in OAuth callback: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)