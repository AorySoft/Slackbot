# SlackBot Q&A Assistant

A Slack integration that uses advanced language models and vector search to provide intelligent answers to user questions, with a feedback system for continuous improvement.

![Slack Bot Banner](https://via.placeholder.com/800x200?text=SlackBot+Q%26A+Assistant)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
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

- **Intelligent Q&A**: Answers questions using the DeepSeek LLM via Groq API
- **Dual Vector Search**: Uses two FAISS indexes to search for relevant information
  - Regular index for AI-generated responses
  - Improved index for human-verified answers
- **Feedback System**: Users can flag incorrect answers with a thumbs-down reaction
- **Admin Dashboard**: Review and improve flagged answers
- **Similarity Detection**: Prevents answering previously flagged questions
- **Content Moderation**: Filters out inappropriate questions

## 🏗️ Architecture

The system is built with a FastAPI backend that handles:

1. Slack event subscriptions
2. LLM interaction via Groq API
3. Vector search using FAISS
4. Database operations for flagged questions
5. Admin dashboard for content moderation

![Architecture Diagram](https://via.placeholder.com/800x400?text=Architecture+Diagram)

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI
- **Database**: SQLite (with SQLAlchemy ORM)
- **Vector Search**: FAISS
- **LLM Provider**: Groq (DeepSeek R1 Distill Llama 70B)
- **Embeddings**: Google Generative AI Embeddings
- **Frontend**: Jinja2 Templates for admin dashboard
- **Integration**: Slack API (Events API, WebClient)

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

1. User reacts with a thumbs-down to an incorrect answer
2. The system stores the question and incorrect answer in the database
3. Admin reviews the flagged question in the dashboard
4. Admin provides the correct answer
5. The corrected Q&A pair is added to the improved FAISS index for future use

## 📥 Installation & Setup

### Prerequisites

- Python 3.8+
- Slack workspace with admin access
- Groq API key
- Google AI API key

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/slackbot-qa-assistant.git
   cd slackbot-qa-assistant
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
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key
   SLACK_BOT_TOKEN=your_slack_bot_token
   SLACK_SIGNING_SECRET=your_slack_signing_secret
   SLACK_CHANNEL_ID=your_slack_channel_id
   ```

5. Initialize the database:
   ```bash
   python -c "from models import Base, engine; Base.metadata.create_all(bind=engine)"
   ```

6. Create initial FAISS indexes:
   ```bash
   python create_initial_indexes.py
   ```

## ⚙️ Configuration

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

### Language Model Configuration

The bot uses the DeepSeek R1 Distill Llama 70B model via Groq. You can modify the model settings in `main.py`:

```python
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="deepseek-r1-distill-llama-70b",  # Can be changed to other models
    temperature=0,  # Adjust for more/less creative responses
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
```

## 🚀 Usage

### Starting the Server

Run the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

For production, consider using Gunicorn with Uvicorn workers:

```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8000
```

### Testing the Bot

Use the following endpoints to test the bot:

- `/test_bot` - Test if the bot can post messages to Slack
- `/test_events` - Test if the events subscription is working
- `/test_event_subscription` - Test if Slack events are reaching the server

### Using the Bot in Slack

The bot will automatically respond to messages in channels it's invited to. For best results:

1. Ask direct questions
2. Use thumbs-down reactions to flag incorrect answers
3. Wait for admin review of flagged questions

## 🖥️ Admin Dashboard

The admin dashboard allows you to:

1. Review flagged questions
2. Provide correct answers
3. Monitor system performance

Access the dashboard at: `http://your-server-url.com/dashboard`

## 🧪 Development

### Adding New Features

To add new features:

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Submit a pull request

### Improving the Model

You can improve the model's performance by:

1. Adding more examples to the knowledge base
2. Adjusting the LLM prompt in `get_llm_response` function
3. Fine-tuning the vector search parameters

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Built with ❤️ by [Your Name/Organization]
