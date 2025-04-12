# SlackBot Q&A Assistant

A Slack integration that uses advanced language models and vector search to provide intelligent answers to user questions, with a feedback system for continuous improvement.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [How It Works](#how-it-works)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Admin Dashboard](#admin-dashboard)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This SlackBot Q&A Assistant is designed to provide intelligent responses to user questions in Slack channels. It uses a combination of LLM (Large Language Model) technology and vector search to find the most relevant answers from a knowledge base. The system includes a feedback mechanism where users can flag incorrect answers, which are then reviewed and improved through an admin dashboard.

## ✨ Features

- **Intelligent Q&A**: Answers questions using Groq's language models
- **Dual Vector Search**: Uses two FAISS indexes to search for relevant information
  - Regular index for AI-generated responses
  - Improved index for human-verified answers
- **Feedback System**: Users can flag incorrect answers with a 👎 reaction
- **Admin Dashboard**: Review and improve flagged answers
- **Similarity Detection**: Prevents answering previously flagged questions
- **Content Moderation**: Filters out inappropriate questions
- **Rate Limiting**: Prevents abuse of the bot
- **Circuit Breaker**: Handles API failures gracefully
- **Metrics & Monitoring**: Tracks bot performance and errors

## 📸 Screenshots

### User Interface
![User Query](screenshots/userQuery.PNG)
*User asking a question in Slack*

![Bot Response](screenshots/botResponseToQuery.PNG)
*Bot providing an answer*

### Feedback System
![Flagging Response](screenshots/flaggingBotResponse.PNG)
*User flagging an incorrect answer with 👎 reaction*

### Admin Dashboard
![Empty Dashboard](screenshots/dashboardWthNoQues.PNG)
*Admin dashboard with no flagged questions*

![Flagged Questions](screenshots/answeringFlaggedQuestions.PNG)
*Admin reviewing and answering flagged questions*

![Updated Answer](screenshots/botUpdatedAnswer.PNG)
*Bot providing an improved answer after admin review*

### Knowledge Base Management
![Upload CSV](screenshots/uploadCSV.PNG)
*Uploading a CSV file to upgrade the knowledge base*

![CSV Preview](screenshots/csvPreview.PNG)
*Preview of the CSV file content in question-answer format*

## 🏗️ Architecture

The system is built with a FastAPI backend that handles:

1. Slack event subscriptions
2. LLM interaction via Groq API
3. Vector search using FAISS
4. Database operations for flagged questions
5. Admin dashboard for content moderation
6. Rate limiting and circuit breaking
7. Metrics collection and monitoring

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI
- **Database**: SQLite (with SQLAlchemy ORM)
- **Vector Search**: FAISS
- **LLM Provider**: Groq
- **Embeddings**: Google Generative AI Embeddings
- **Frontend**: Jinja2 Templates for admin dashboard
- **Integration**: Slack API (Events API, WebClient)
- **Monitoring**: Custom metrics system
- **Caching**: TTLCache for message processing
- **Rate Limiting**: Custom rate limiter
- **Circuit Breaking**: Custom circuit breaker

## 🧠 How It Works

### Question Answering Flow

1. User asks a question in a Slack channel
2. The bot receives the message via Slack Events API
3. The question is processed:
   - Checked against flagged content
   - Embeddings are generated for vector search
   - Both FAISS indexes are queried for relevant information
4. The LLM generates a response with context from:
   - Human-verified answers (prioritized)
   - AI-generated answers (fallback)
5. The response is posted back to the Slack thread

### Feedback Flow

1. User reacts with 👎 to an incorrect answer
2. The system stores the question in the database
3. Admin reviews the flagged question in the dashboard
4. Admin provides the correct answer
5. The corrected Q&A pair is added to the improved FAISS index for future use

## 📥 Installation & Setup

### Prerequisites

- Python 3.8+
- Slack workspace with admin access
- Groq API key
- Google API key for embeddings

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/AorySoft/Slackbot.git
   cd SLACK_BOT
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following variables:
   ```
   # API Keys
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key  # Required for text embeddings using Google's Generative AI

   # Slack Configuration
   SLACK_BOT_TOKEN=xoxb-your-bot-token
   SLACK_SIGNING_SECRET=your-signing-secret
   SLACK_APP_TOKEN=xapp-your-app-token
   SLACK_CHANNEL_ID=your-channel-id

   # Application Configuration
   APP_HOST=0.0.0.0
   APP_PORT=8000
   ```

## ⚙️ Configuration

### Environment Variables

The application uses the following environment variables:

- **API Keys**
  - `GROQ_API_KEY`: Your Groq API key for accessing LLM
  - `GOOGLE_API_KEY`: Your Google API key for text embeddings using Google's Generative AI (model: models/embedding-001)

- **Slack Configuration**
  - `SLACK_BOT_TOKEN`: Bot token starting with `xoxb-`
  - `SLACK_SIGNING_SECRET`: Used to verify requests from Slack
  - `SLACK_APP_TOKEN`: App token starting with `xapp-`
  - `SLACK_CHANNEL_ID`: ID of the Slack channel the bot should monitor

- **Application Configuration**
  - `APP_HOST`: Host address to bind the server (default: 0.0.0.0)
  - `APP_PORT`: Port to run the server on (default: 8000)

### Slack App Configuration

1. Create a new Slack app at [api.slack.com](https://api.slack.com/apps)
2. Enable the following OAuth scopes:
   - `channels:history`
   - `channels:read`
   - `chat:write`
   - `reactions:read`
3. Install the app to your workspace
4. Enable Event Subscriptions:
   - Set Request URL to `https://your-server-url.com/slack/events`
   - Subscribe to the following events:
     - `message.channels`
     - `reaction_added`
5. Copy the Bot User OAuth Token and Signing Secret to your `.env` file

## 🚀 Usage

### Starting the Server

Run the FastAPI server:

```bash
uvicorn main:app --host ${APP_HOST} --port ${APP_PORT} --reload
```

For production, consider using Gunicorn with Uvicorn workers:

```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4 --bind ${APP_HOST}:${APP_PORT}
```

### Testing the Bot

Use the following endpoints to test the bot:

- `/test_bot` - Test if the bot can post messages to Slack
- `/test_events` - Test if the events subscription is working
- `/test_event_subscription` - Test if Slack events are reaching the server
- `/health` - Check the health status of the bot

### Using the Bot in Slack

The bot will automatically respond to messages in channels it's invited to. For best results:

1. Ask direct questions
2. Use 👎 reactions to flag incorrect answers
3. Wait for admin review of flagged questions

## 🖥️ Admin Dashboard

The admin dashboard allows you to:

1. Review flagged questions
2. Provide correct answers
3. Reject inappropriate questions

Access the dashboard at: `http://your-server-url.com/dashboard`

## 🧪 Development

### Adding New Features

To add new features:

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Submit a pull request

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

