# AI Powered Slack Chatbot

## Overview  
This is a **Slack chatbot** that listens to messages in a channel (when tagged), sends queries to the LLM (GPT-4 in this case), and replies with generated responses. It also stores the last 5 messages and sends them as context for better replies.
---

## Tech Stack  
- **FastAPI** - Backend framework  
- **Slack SDK** - Handles Slack bot interactions  
- **OpenAI API** - Generates responses using GPT-4  
- **SQLite** - Stores conversation history  
- **ngrok** - Exposes FastAPI server for local testing  
- **Railway** - Deployment  
---

## Architecture
![chatbot-architecture](https://github.com/user-attachments/assets/196735fb-93c8-41b6-91d6-8c85d9e8d41a)

## Installation Guide 🛠️  


### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/suyashk15/slack-chatbot.git
cd slack-chatbot
```
### 2️⃣ Set Up a Virtual Environment 
```bash
python -m venv env
cd env\Scripts\     # Windows
activate
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Set Up Environment Variables
- Create a .env file in the project root and add:
```bash
SLACK_BOT_TOKEN=
SLACK_SIGNING_SECRET=
OPENAI_API_KEY=
SLACK_CLIENT_ID=
SLACK_CLIENT_SECRET=
SLACK_OAUTH_REDIRECT_URI={Ngrok HTTPS URL}/slack/oauth/callback
SLACK_APP_ID=
```
- Place your credentials from Slack and OpenAI.

## Running the App Locally  

### 1️⃣ Start the FastAPI Server
```bash
uvicorn main:app --reload --port 8000
```
### 2️⃣ Expose It Using ngrok
```bash
ngrok http 8000
```
Copy the ngrok HTTPS URL (e.g., https://random-string.ngrok.io) and use it for Slack event subscriptions.

## Direct Slack Installation (Available for limited time)  

### 1️⃣ Click on this link: [Get Droid Assistant](https://bit.ly/4b5m4T1)

### 2️⃣ Select the workspace and allow permissions (Admin permission required)

### 3️⃣ Add app to the channel (using /invite @Droid Assistant) and start chatting.
---
## Demo Video  

https://github.com/user-attachments/assets/13d3bcb2-ad9e-4979-a258-98e20f265830


