import os
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from slack_sdk import WebClient
import logging
import hmac
import hashlib
import string
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from sqlalchemy.orm import Session
import models
import schemas
from models import get_db
from typing import List, Dict
from uuid import uuid4
from langchain_core.documents import Document
import json
import time

# Load environment variables
load_dotenv()

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Debug: Print environment variables
print("=== Debug: Environment Variables ===")
print(f"GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY')}")
print(f"GROQ_API_KEY: {os.getenv('GROQ_API_KEY')}")
print(f"SLACK_SIGNING_SECRET starts with: {os.getenv('SLACK_SIGNING_SECRET')[:5] if os.getenv('SLACK_SIGNING_SECRET') else 'NOT SET'}")
print(f"SLACK_BOT_TOKEN starts with: {os.getenv('SLACK_BOT_TOKEN')[:10] if os.getenv('SLACK_BOT_TOKEN') else 'NOT SET'}")
print("=================================")

# Initialize FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize Slack client
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

# Initialize embeddings and FAISS indexes
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
faiss_index_improved = FAISS.load_local("faiss_index_improved", embeddings, allow_dangerous_deserialization=True)

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Get bot's user ID
try:
    BOT_ID = slack_client.auth_test()['user_id']
    print(f"\n=== Bot Initialization ===")
    print(f"Bot ID: {BOT_ID}")
    print("==========================")
    logger.info(f"Bot ID: {BOT_ID}")
except Exception as e:
    logger.error(f"Failed to get bot ID: {e}")
    BOT_ID = None

# Global state
message_counts = {}
welcome_messages = {}
# BAD_WORDS = ['hmm', 'no', 'bad']  # Customize this list as needed
processed_messages = set()  # Set to track processed message IDs

# class WelcomeMessage:
#     START_TEXT = {
#         'type': 'section',
#         'text': {
#             'type': 'mrkdwn',
#             'text': (
#                 'Welcome to this awesome channel! \n\n'
#                 '*Get started by completing the tasks!*'
#             )
#         }
#     }

#     DIVIDER = {'type': 'divider'}

#     def __init__(self, channel):
#         self.channel = channel
#         self.icon_emoji = ':robot_face:'
#         self.timestamp = ''
#         self.completed = False

#     def get_message(self):
#         return {
#             'ts': self.timestamp,
#             'channel': self.channel,
#             'username': 'Welcome Robot!',
#             'icon_emoji': self.icon_emoji,
#             'blocks': [
#                 self.START_TEXT,
#                 self.DIVIDER,
#                 self._get_reaction_task()
#             ]
#         }

#     def _get_reaction_task(self):
#         checkmark = ':white_check_mark:' if self.completed else ':white_large_square:'
#         text = f'{checkmark} *React to this message!*'
#         return {'type': 'section', 'text': {'type': 'mrkdwn', 'text': text}}

# def send_welcome_message(channel, user):
#     """Send a welcome message to a user in a channel"""
#     if channel not in welcome_messages:
#         welcome_messages[channel] = {}

#     if user in welcome_messages[channel]:
#         return

#     welcome = WelcomeMessage(channel)
#     message = welcome.get_message()
#     try:
#         response = slack_client.chat_postMessage(**message)
#         welcome.timestamp = response['ts']
#         welcome_messages[channel][user] = welcome
#         logger.info(f"Sent welcome message to user {user} in channel {channel}")
#     except Exception as e:
#         logger.error(f"Failed to send welcome message: {e}")

# def check_if_bad_words(message: str) -> bool:
#     """Check if message contains any bad words"""
#     msg = message.lower()
#     msg = msg.translate(str.maketrans('', '', string.punctuation))
#     return any(word in msg for word in BAD_WORDS)

def verify_slack_signature(request_body: str, timestamp: str, signature: str) -> bool:
    """Verify the request signature from Slack"""
    # Form the base string by combining version, timestamp, and request body
    sig_basestring = f"v0:{timestamp}:{request_body}"
    
    # Calculate a new signature using your signing secret
    my_signature = 'v0=' + hmac.new(
        os.getenv('SLACK_SIGNING_SECRET').encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Compare the signatures
    return hmac.compare_digest(my_signature, signature)

async def get_llm_response(text: str, db: Session) -> str:
    """Get response from LLM with context from both FAISS indexes"""
    try:
        print("\n=== Starting LLM Response Function ===")
        
        # Step 1: Get relevant documents from both FAISS indexes
        print("\nStep 1: Attempting to retrieve documents from FAISS indexes...")
        try:
            # Query both indexes
            regular_docs = faiss_index.similarity_search(text, k=2)
            improved_docs = faiss_index_improved.similarity_search(text, k=2)
            
            print(f"Retrieved {len(regular_docs)} regular docs and {len(improved_docs)} improved docs")
            
            # Prepare context, prioritizing improved docs
            context_parts = []
            
            if improved_docs:
                context_parts.append("\n=== HUMAN VERIFIED ANSWERS (USE THESE FIRST!) ===")
                for i, doc in enumerate(improved_docs, 1):
                    context_parts.append(f"Verified Answer {i}: {doc.page_content}")
            
            if regular_docs:
                context_parts.append("\n=== AI GENERATED ANSWERS (Only use if verified answers don't help) ===")
                for i, doc in enumerate(regular_docs, 1):
                    context_parts.append(f"AI Answer {i}: {doc.page_content}")
            
            context = "\n".join(context_parts)
            
            print("Context preview:", context[:200])
            
        except Exception as faiss_error:
            print(f"Error retrieving from FAISS: {faiss_error}")
            raise

        # Step 2: Create chat prompt template
        print("\nStep 2: Creating chat prompt template...")
        try:
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """Listen carefully! You have answers from TWO DIFFERENT DATABASES:

1. FROM FAISS_INDEX_IMPROVED (HUMAN VERIFIED DATABASE):
{improved_answers}

2. FROM FAISS_INDEX (REGULAR DATABASE):
{regular_answers}

IMPORTANT RULES:
- If you find the same answer in both databases, ALWAYS USE THE ONE FROM FAISS_INDEX_IMPROVED!
- FAISS_INDEX_IMPROVED answers are human-verified and 100% accurate
- FAISS_INDEX answers are AI-generated and less reliable

Step by step how to answer:
1. First, look at FAISS_INDEX_IMPROVED answers
2. If you find a relevant answer there, USE IT and mention "Based on verified answer from FAISS_INDEX_IMPROVED:"
3. Only if you don't find anything in FAISS_INDEX_IMPROVED, check FAISS_INDEX
4. If using FAISS_INDEX, say "Based on AI-generated answer from FAISS_INDEX:"
5. If nothing relevant in either database, say "No relevant answers found in either database" and answer from your knowledge

Remember: FAISS_INDEX_IMPROVED > FAISS_INDEX. Always prioritize improved index!"""
                ),
                ("human", "{question}")
            ])
            print("Chat prompt template created successfully")
        except Exception as prompt_error:
            print(f"Error creating prompt template: {prompt_error}")
            raise

        # Step 3: Create and invoke chain
        print("\nStep 3: Creating and invoking chain...")
        try:
            chain = prompt | llm
            print("Chain created, attempting to invoke...")
            
            # Prepare separate context strings for each index
            improved_answers = "No verified answers found."
            if improved_docs:
                improved_answers = "\n".join([f"Answer {i+1}: {doc.page_content}" 
                                           for i, doc in enumerate(improved_docs)])

            regular_answers = "No AI-generated answers found."
            if regular_docs:
                regular_answers = "\n".join([f"Answer {i+1}: {doc.page_content}" 
                                          for i, doc in enumerate(regular_docs)])
            
            response = chain.invoke({
                "improved_answers": improved_answers,
                "regular_answers": regular_answers,
                "question": text
            })
            print("Chain invoked successfully")
            print("Response preview:", str(response)[:100])
            
            return response.content
            
        except Exception as chain_error:
            print(f"Error in chain execution: {chain_error}")
            raise

        print("\n=== LLM Response Function Completed Successfully ===")
        return response.content

    except Exception as e:
        print(f"\n!!! ERROR in get_llm_response: {str(e)}")
        logger.error(f"Detailed error in get_llm_response: {str(e)}", exc_info=True)
        return f"I apologize, but I encountered an error: {str(e)}"

def store_flagged_question(question: str, db: Session):
    """Store a flagged question in the database"""
    try:
        db_question = models.FlaggedQuestion(question=question)
        db.add(db_question)
        db.commit()
        db.refresh(db_question)
        logger.info(f"Stored flagged question: {question}")
        return db_question
    except Exception as e:
        logger.error(f"Error storing flagged question: {e}")
        db.rollback()
        raise

def get_flagged_questions(db: Session) -> List[schemas.FlaggedQuestion]:
    """Get all unanswered flagged questions"""
    try:
        questions = db.query(models.FlaggedQuestion).filter(
            models.FlaggedQuestion.is_answered == False
        ).all()
        return questions
    except Exception as e:
        logger.error(f"Error getting flagged questions: {e}")
        return []

@app.get("/")
async def test_endpoint():
    """Test endpoint to verify server is running"""
    logger.info("Test endpoint was called!")
    return {"status": "Server is running!"}

@app.post("/slack/events")
async def slack_events(request: Request):
    """Handle Slack events"""
    print("\n=== Received Slack Event ===")
    print(f"Time: {datetime.now().isoformat()}")
    
    try:
        # Log all headers
        headers = dict(request.headers)
        print("\nHeaders:")
        for key, value in headers.items():
            print(f"{key}: {value}")
        
        # Get and log raw body
        raw_body = await request.body()
        body_str = raw_body.decode()
        print(f"\nRaw Body: {body_str}")
        
        # Verify Slack signature
        timestamp = headers.get('x-slack-request-timestamp', '')
        signature = headers.get('x-slack-signature', '')
        
        print(f"\n=== Signature Verification ===")
        print(f"Timestamp: {timestamp}")
        print(f"Received Signature: {signature}")
        
        # Check if timestamp is too old
        if abs(time.time() - int(timestamp)) > 60 * 5:
            print("❌ Request timestamp is too old")
            return {"error": "Invalid timestamp"}
            
        # Verify signature
        is_valid = verify_slack_signature(body_str, timestamp, signature)
        print(f"Signature Valid: {is_valid}")
        
        if not is_valid:
            print("❌ Invalid signature")
            return {"error": "Invalid signature"}
        
        print("✅ Signature verification passed")
        
        # Parse and log JSON body
        try:
            body = await request.json()
            print(f"\nParsed JSON body: {body}")
            
            # Handle URL verification
            if body.get("type") == "url_verification":
                challenge = body.get("challenge")
                print(f"Returning challenge: {challenge}")
                return {"challenge": challenge}
            
            # Log event details
            event = body.get("event", {})
            event_type = event.get("type")
            print(f"\nEvent type: {event_type}")
            print(f"Full event details: {event}")
            
            # Handle message events
            if event_type == "message":
                channel_id = event.get('channel')
                user_id = event.get('user')
                text = event.get('text', '')
                bot_id = event.get('bot_id')
                message_id = event.get('client_msg_id', '')  # Get message ID
                
                print(f"\n=== Message Details ===")
                print(f"Channel: {channel_id}")
                print(f"User: {user_id}")
                print(f"Text: {text}")
                print(f"Bot ID: {bot_id}")
                print(f"Message ID: {message_id}")
                print("=======================")
                
                # Skip if message is from a bot or is our own message
                if bot_id or user_id == BOT_ID:
                    print("Skipping bot message")
                    return {"ok": True}
                
                # Skip if we've already processed this message
                if message_id in processed_messages:
                    print(f"Message {message_id} already processed, skipping")
                    return {"ok": True}
                
                # Process user message
                if text and user_id and message_id:  # Only process if we have a message ID
                    try:
                        db = next(get_db())
                        llm_response = await get_llm_response(text, db)
                        
                        # Send response
                        response = slack_client.chat_postMessage(
                            channel=channel_id,
                            thread_ts=event.get('ts'),
                            text=llm_response
                        )
                        
                        # Add message ID to processed set
                        processed_messages.add(message_id)
                        print(f"✅ Added message {message_id} to processed set")
                        print("✅ Sent response successfully:", response)
                    except Exception as e:
                        print(f"❌ Error sending response: {str(e)}")
                        logger.error(f"Error sending response: {str(e)}", exc_info=True)
            
            # Handle reaction events
            elif event_type == "reaction_added":
                # Skip if reaction is from the bot itself
                if event.get('user') == BOT_ID:
                    print("Skipping reaction from bot")
                    return {"ok": True}
                    
                if event.get('reaction') == '-1':  # Check for thumbs down reaction
                    try:
                        db = next(get_db())
                        # Get the message that was reacted to
                        result = slack_client.conversations_history(
                            channel=event.get('item', {}).get('channel'),
                            latest=event.get('item', {}).get('ts'),
                            limit=1,
                            inclusive=True
                        )
                        
                        if result['messages']:
                            # Get the thread of the message to find both question and answer
                            thread_result = slack_client.conversations_replies(
                                channel=event.get('item', {}).get('channel'),
                                ts=result['messages'][0].get('thread_ts', result['messages'][0].get('ts')),
                                limit=2  # Get both the question and the bot's response
                            )
                            
                            if thread_result['messages'] and len(thread_result['messages']) >= 2:
                                user_question = thread_result['messages'][0].get('text', '')  # First message is user's question
                                bot_response = thread_result['messages'][1].get('text', '')   # Second message is bot's response
                                
                                print(f"\n=== Storing Disliked Q&A Pair ===")
                                print(f"User Question: {user_question}")
                                print(f"Bot Response: {bot_response}")
                                
                                # Store both question and bot's response
                                db_question = models.FlaggedQuestion(
                                    question=user_question,
                                    llm_response=bot_response,  # This is the actual LLM response
                                    dislike_count=1
                                )
                                db.add(db_question)
                                db.commit()
                                print("✅ Successfully stored disliked Q&A pair")
                    except Exception as e:
                        print(f"❌ Error handling reaction: {str(e)}")
                        logger.error(f"Error handling reaction: {str(e)}", exc_info=True)
            
            return {"ok": True}
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {str(e)}")
            logger.error(f"Error parsing JSON: {str(e)}", exc_info=True)
            return {"error": "Invalid JSON"}
            
    except Exception as e:
        print(f"❌ Error processing event: {str(e)}")
        logger.error(f"Error processing event: {str(e)}", exc_info=True)
        return {"error": str(e)}

@app.get("/test_events")
async def test_events():
    """Test if events endpoint is accessible"""
    try:
        print("\n=== Testing Events Endpoint ===")
        channel_id = os.getenv("SLACK_CHANNEL_ID")
        print(f"Posting to channel: {channel_id}")
        
        # Send a test message
        response = slack_client.chat_postMessage(
            channel=channel_id,
            text="🔍 Testing events... You should see the bot respond to this!"
        )
        
        # Extract only the necessary data from response
        response_data = {
            "ok": response.get("ok", False),
            "channel": response.get("channel"),
            "ts": response.get("ts"),
            "message": response.get("message", {}).get("text", "")
        }
        
        print(f"Test message sent: {response_data}")
        return {
            "status": "success",
            "message": "Test message sent, check your Slack channel and server logs",
            "response": response_data
        }
    except Exception as e:
        print(f"❌ Error testing events: {str(e)}")
        return {"status": "error", "error": str(e)}

# @app.get("/message-count/{user_id}")
# async def get_message_count(user_id: str):
#     """Get message count for a specific user"""
#     count = message_counts.get(user_id, 0)
#     return {"user_id": user_id, "message_count": count}

# @app.get("/test_message")
# async def test_message():
#     """Test sending a message to verify Slack token"""
#     try:
#         channel_id = os.getenv("SLACK_CHANNEL_ID")
#         response = slack_client.chat_postMessage(
#             channel=channel_id,
#             text="Test message from bot!"
#         )
#         logger.info(f"Test message response: {response}")
        
#         # Return simplified response
#         return {
#             "status": "success",
#             "message_sent": True,
#             "channel": response.get("channel"),
#             "timestamp": response.get("ts")
#         }
#     except Exception as e:
#         logger.error(f"Error sending test message: {e}")
#         return {
#             "status": "error",
#             "message_sent": False,
#             "error": str(e)
#         }

# @app.post("/test_webhook")
# async def test_webhook(request: Request):
#     """Test endpoint to verify webhook is accessible"""
#     body = await request.body()
#     headers = dict(request.headers)
#     logger.info("----------------------------------------")
#     logger.info("TEST WEBHOOK CALLED")
#     logger.info(f"Headers: {headers}")
#     logger.info(f"Body: {body.decode()}")
#     logger.info("----------------------------------------")
#     return {"status": "received"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Display the dashboard of flagged questions"""
    questions = get_flagged_questions(db)
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "questions": questions}
    )

@app.post("/submit_answer")
async def submit_answer(
    answer_data: schemas.AnswerCreate,
    db: Session = Depends(get_db)
):
    """Handle submission of answers to flagged questions"""
    try:
        # Get the question from database
        question = db.query(models.FlaggedQuestion).filter(
            models.FlaggedQuestion.id == answer_data.question_id
        ).first()
        
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")
        
        # Update question with correct answer
        question.correct_answer = answer_data.correct_answer
        question.is_answered = True
        
        # Create combined text for embedding
        combined_text = f"""Question: {question.question}
Answer: {answer_data.correct_answer}"""
        
        # Create Document object
        document = Document(
            page_content=combined_text,
            metadata={
                "source": "human_verified",
                "question_id": str(question.id),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Generate UUID for the document
        doc_uuid = str(uuid4())
        
        # Store in improved FAISS index
        try:
            print(f"Adding to improved index with UUID {doc_uuid}:")
            print(f"Content: {combined_text}")
            print(f"Metadata: {document.metadata}")
            
            # Add document to FAISS
            faiss_index_improved.add_documents(documents=[document], ids=[doc_uuid])
            
            # Save the updated index
            faiss_index_improved.save_local("faiss_index_improved")
            
            # Update database
            question.embedding_id = doc_uuid
            db.commit()
            
            logger.info(f"Stored answer and updated FAISS index for question ID {answer_data.question_id}")
            return {"status": "success"}
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating FAISS index: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        logger.error(f"Error storing answer: {e}")
        db.rollback()
        return {"status": "error", "message": str(e)}

# endpoint to handle dislikes
@app.post("/record_dislike/{question_id}")
async def record_dislike(
    question_id: int,
    db: Session = Depends(get_db)
):
    """Record a dislike for a question/answer pair"""
    try:
        question = db.query(models.FlaggedQuestion).filter(
            models.FlaggedQuestion.id == question_id
        ).first()
        
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")
        
        question.dislike_count += 1
        db.commit()
        
        return {"status": "success", "dislike_count": question.dislike_count}
    except Exception as e:
        db.rollback()
        logger.error(f"Error recording dislike: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/test_bot")
async def test_bot():
    """Test if bot can post messages"""
    try:
        print("\n=== Testing Bot Message ===")
        channel_id = os.getenv("SLACK_CHANNEL_ID")
        print(f"Posting to channel: {channel_id}")
        
        response = slack_client.chat_postMessage(
            channel=channel_id,
            text="🔍 Bot test message - checking if I can post to this channel!"
        )
        
        print(f"Response from Slack: {response}")
        return {"status": "success", "response": response}
    except Exception as e:
        print(f"❌ Error testing bot: {str(e)}")
        return {"status": "error", "error": str(e)}

@app.post("/test_event_subscription")
async def test_event_subscription(request: Request):
    """Test endpoint to verify Slack events are reaching the server"""
    print("\n=== Test Event Subscription ===")
    
    # Get headers
    headers = dict(request.headers)
    print("Headers received:", headers)
    
    # Get body
    body = await request.body()
    body_str = body.decode()
    print("Body received:", body_str)
    
    try:
        # Parse JSON body
        json_body = await request.json()
        print("Parsed JSON:", json_body)
        
        return {
            "status": "success",
            "message": "Event received and logged",
            "event_type": json_body.get("type"),
            "event": json_body.get("event", {})
        }
    except Exception as e:
        print(f"Error processing event: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    logger.info("Starting server with DEBUG logging...")
    logger.info(f"Bot token starts with: {os.getenv('SLACK_BOT_TOKEN')[:15]}...")
    logger.info(f"Channel ID: {os.getenv('SLACK_CHANNEL_ID')}")
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    ) 