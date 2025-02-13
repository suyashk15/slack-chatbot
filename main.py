from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.signature import SignatureVerifier
from openai import AsyncOpenAI
import sqlite3
import aiosqlite
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict
import json
import requests
from fastapi.responses import RedirectResponse
import httpx
from urllib.parse import urlencode

load_dotenv()

# Database setup
DB_NAME = "slack_messages.db"

def init_db():
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    init_db()
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Initialize clients
slack_client = AsyncWebClient(token=os.environ["SLACK_BOT_TOKEN"])
signature_verifier = SignatureVerifier(os.environ["SLACK_SIGNING_SECRET"])

SLACK_CLIENT_ID = os.getenv("SLACK_CLIENT_ID", "placeholder-client-id")
SLACK_CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET", "placeholder-client-secret")
SLACK_OAUTH_REDIRECT_URI = os.getenv("SLACK_OAUTH_REDIRECT_URI", "http://localhost:8000/slack/oauth/callback")
SLACK_APP_ID = os.getenv("SLACK_APP_ID", "placeholder-app-id")

try:
    openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=30.0
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    raise

async def get_conversation_history(channel_id: str) -> List[Dict]:
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
        return [dict(msg) for msg in reversed(messages)]

async def store_message(channel: str, user: str, text: str):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            """
            INSERT INTO messages (channel, user_id, message)
            VALUES (?, ?, ?)
            """,
            (channel, user, text)
        )
        await db.commit()

# Verify the request is from Slack
@app.post("/")
async def slack_events(request: Request):
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    print(f"Received event with timestamp: {timestamp}")

    if not signature_verifier.is_valid(
        body=body,
        timestamp=timestamp,
        signature=signature
    ):
        raise HTTPException(status_code=400, detail="Invalid request signature")

    event_data = json.loads(body)

    print(f"Event data: {event_data}")

    if event_data.get("type") == "url_verification":
        return {"challenge": event_data["challenge"]}

    event = event_data.get("event", {})
    if event.get("type") == "app_mention":
        channel_id = event["channel"]
        user_id = event["user"]
        text = event["text"]
        thread_ts = event.get("thread_ts", event["ts"])

        print(f"Processing mention in channel: {channel_id}") 

        await store_message(channel_id, user_id, text)
        previous_messages = await get_conversation_history(channel_id)

        try:
            # Check if the bot is in the channel
            try:
                channel_info = await slack_client.conversations_info(channel=channel_id)
                print(f"Channel info: {channel_info}")
            except Exception as e:
                print(f"Error getting channel info: {e}")
                # Try to join the channel
                try:
                    join_response = await slack_client.conversations_join(channel=channel_id)
                    if not join_response.get("ok"):
                        raise Exception(f"Failed to join channel: {join_response.get('error')}")
                    print(f"Joined channel: {channel_id}")
                except Exception as join_error:
                    print(f"Error joining channel: {join_error}")
                    await slack_client.chat_postMessage(
                        channel=user_id,  # Send a DM to the user
                        text=f"Sorry, I couldn't join the channel <#{channel_id}>. Please add me to the channel and try again."
                    )
                    return {"ok": True}

            # Format conversation history for the LLM
            conversation_history = "\n".join([
                f"User: {msg['message']}" for msg in previous_messages
            ])

            # Call OpenAI
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

            # Send response back to Slack
            try:
                await slack_client.chat_postMessage(
                    channel=channel_id,
                    text=response.choices[0].message.content,
                    thread_ts=thread_ts
                )
                print(f"Successfully sent message to channel: {channel_id}")
            except Exception as post_error:
                print(f"Error posting message: {post_error}")
                await slack_client.chat_postMessage(
                    channel=user_id,  # Send a DM to the user
                    text="Sorry, I encountered an error posting your message."
                )

        except Exception as e:
            print(f"Error processing message: {e}")
            await slack_client.chat_postMessage(
                channel=user_id,  # Send a DM to the user
                text="Sorry, I encountered an error processing your request."
            )

    return {"ok": True}


@app.get("/slack/install")
async def slack_install():
    """
    Direct install URL endpoint that redirects to Slack's OAuth authorization.
    """
    params = {
        "client_id": SLACK_CLIENT_ID,
        "scope": "app_mentions:read,channels:history,channels:join,chat:write,groups:read,im:read,mpim:read,channels:read",
        "user_scope": "chat:write",
        "redirect_uri": SLACK_OAUTH_REDIRECT_URI
    }

    auth_url = f"https://slack.com/oauth/v2/authorize?{urlencode(params)}"
    return RedirectResponse(url=auth_url, status_code=302)


@app.get("/slack/oauth/callback")
async def slack_oauth_callback(code: str):
    """
    Handles the OAuth callback from Slack and exchanges the authorization code for an access token.
    """
    async with httpx.AsyncClient() as client:
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
        return {"error": "OAuth failed", "details": data}

    return RedirectResponse(url=f"https://slack.com/app_redirect?app={SLACK_APP_ID}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)