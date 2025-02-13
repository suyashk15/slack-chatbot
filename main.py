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

    if not signature_verifier.is_valid(
        body=body,
        timestamp=timestamp,
        signature=signature
    ):
        raise HTTPException(status_code=400, detail="Invalid request signature")

    # Parse the event data
    event_data = json.loads(body)

    # Handle Slack URL verification
    if event_data.get("type") == "url_verification":
        return {"challenge": event_data["challenge"]}

    # Process the event
    event = event_data.get("event", {})
    if event.get("type") == "app_mention":
        channel_id = event["channel"]
        user_id = event["user"]
        text = event["text"]
        thread_ts = event.get("thread_ts", event["ts"])

        # Store the current message
        await store_message(channel_id, user_id, text)

        # Get conversation history
        previous_messages = await get_conversation_history(channel_id)

        # Format conversation history for the LLM
        conversation_history = "\n".join([
            f"User: {msg['message']}" for msg in previous_messages
        ])

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant in a Slack channel. Keep responses concise and relevant."
                    },
                    {
                        "role": "user",
                        "content":  f"""Previous conversation:{conversation_history}
                                        Current message: {text}
                                        Please provide a helpful response:"""
                    }
                ]
            )

            # Send response back to Slack
            await slack_client.chat_postMessage(
                channel=channel_id,
                text=response.choices[0].message.content,
                thread_ts=thread_ts
            )

        except Exception as e:
            print(f"Error: {e}")
            await slack_client.chat_postMessage(
                channel=channel_id,
                text="Sorry, I encountered an error processing your request.",
                thread_ts=thread_ts
            )

    return {"ok": True}

# Exchange the code for an access token
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

    access_token = data["access_token"]
    team_id = data["team"]["id"]

    # You may want to store the access_token in a database for later use
    return RedirectResponse(url="https://slack.com/app_redirect?team=" + team_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)