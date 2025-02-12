# AI Powered Slack Chatbot

## Overview  
This is a **Slack chatbot** that listens to messages in a channel (when tagged), sends queries to the LLM (GPT-4 in this case), and replies with generated responses. It also stores the last 5 messages and sends them as context for better replies.
---

## Features  
- ✅ Listens for messages where it's tagged in a Slack channel  
- ✅ Uses OpenAI (GPT-4) to generate responses  
- ✅ Maintains conversation history (last 5 messages)  
- ✅ Replies in the same thread  
- ✅ Uses FastAPI and SQLite for lightweight storage  
---

## Tech Stack  
- **FastAPI** - Backend framework  
- **Slack SDK** - Handles Slack bot interactions  
- **OpenAI API** - Generates responses using GPT-4  
- **SQLite** - Stores conversation history  
- **ngrok** - Exposes FastAPI server for local testing  
- **Railway** - Deployment  
---

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
```
- Place your credentials from Slack and OpenAI.

## Running the App Locally 🚀  

### 1️⃣ Start the FastAPI Server
```bash
uvicorn main:app --reload --port 8000
```
### 2️⃣ Expose It Using ngrok
```bash
ngrok http 8000
```
Copy the ngrok HTTPS URL (e.g., https://random-string.ngrok.io) and use it for Slack event subscriptions.
