import os
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import UploadFile, File
import csv
from io import StringIO
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
from typing import List, Dict, Tuple
from uuid import uuid4
from langchain_core.documents import Document
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from cachetools import TTLCache
import json
import time
import numpy as np
import re
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from database import get_db, init_db
from cache import (
    get_cached_llm_response, set_cached_llm_response,
    get_cached_embedding, set_cached_embedding,
    is_message_processed, mark_message_processed
)
from rate_limiter import rate_limit_middleware
from monitoring import metrics, metrics_middleware
from circuitbreaker import circuit
from tenacity import retry, stop_after_attempt, wait_exponential
from groq import Groq

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
processed_messages = TTLCache(maxsize=10000, ttl=86400)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# Add metrics middleware
app.middleware("http")(metrics_middleware)

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def verify_slack_signature(request: Request) -> bool:
    """Verify the request signature from Slack"""
    try:
        # Get headers
        timestamp = request.headers.get('x-slack-request-timestamp', '')
        signature = request.headers.get('x-slack-signature', '')
        
        # Check if timestamp is too old
        if abs(time.time() - int(timestamp)) > 60 * 5:
            logger.error("Request timestamp is too old")
            return False
            
        # Get raw body
        body = request.body()
        body_str = body.decode('utf-8')
        
        # Form the base string
        sig_basestring = f"v0:{timestamp}:{body_str}"
        
        # Calculate signature
        my_signature = 'v0=' + hmac.new(
            os.getenv('SLACK_SIGNING_SECRET').encode(),
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(my_signature, signature)
    except Exception as e:
        logger.error(f"Error verifying Slack signature: {str(e)}")
        return False

def is_flagged_question(text: str) -> bool:
    """Check if the given text is asking about a flagged question"""
    try:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a classifier that determines if a user's question is asking about flagged or disliked content.
                Return ONLY the number 1 if the question is asking about flagged/disliked content, or 0 if it's not.
                DO NOT return any other text or explanation."""
            ),
            ("human", "{question}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"question": text})
        return response.content.strip() == "1"
    except Exception as e:
        print(f"Error in is_flagged_question: {e}")
        return False

def get_conversation_history(thread_id: str, db: Session) -> List[Dict[str, str]]:
    """Retrieve conversation history for a thread"""
    try:
        history = db.query(models.ConversationHistory).filter(
            models.ConversationHistory.thread_id == thread_id
        ).first()
        
        if history and history.conversation:
            return json.loads(history.conversation)
        return []
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        return []

def update_conversation_history(thread_id: str, human_msg: str, ai_response: str, db: Session):
    """Update or create conversation history for a thread"""
    try:
        # Get existing history
        history_record = db.query(models.ConversationHistory).filter(
            models.ConversationHistory.thread_id == thread_id
        ).first()
        
        # New exchange
        new_exchange = {"Human": human_msg, "AI": ai_response}
        
        if history_record:
            # Update existing record
            conversation = json.loads(history_record.conversation)
            conversation.append(new_exchange)
            history_record.conversation = json.dumps(conversation)
        else:
            # Create new record
            conversation = [new_exchange]
            new_record = models.ConversationHistory(
                thread_id=thread_id,
                conversation=json.dumps(conversation)
            )
            db.add(new_record)
        
        db.commit()
    except Exception as e:
        logger.error(f"Error updating conversation history: {e}")
        # Don't rollback, just log the error and continue

def find_similar_flagged_questions(text: str, db: Session, threshold: float = 0.8) -> List[Tuple[models.FlaggedQuestion, float]]:
    """Find similar flagged questions using cosine similarity"""
    try:
        # Get embedding for the input text
        query_embedding = embeddings.embed_query(text)
        
        # Get all flagged questions with embeddings
        flagged_questions = db.query(models.FlaggedQuestion).filter(
            models.FlaggedQuestion.question_embedding.isnot(None)
        ).all()
        
        similar_questions = []
        for question in flagged_questions:
            # Convert stored embedding from JSON string to numpy array
            stored_embedding = np.array(json.loads(question.question_embedding))
            query_embedding_np = np.array(query_embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding_np, stored_embedding) / (
                np.linalg.norm(query_embedding_np) * np.linalg.norm(stored_embedding)
            )
            
            if similarity >= threshold:
                similar_questions.append((question, float(similarity)))
        
        # Sort by similarity score and get top 5
        similar_questions.sort(key=lambda x: x[1], reverse=True)
        return similar_questions[:5]
    except Exception as e:
        print(f"Error in find_similar_flagged_questions: {e}")
        return []

@circuit(failure_threshold=5, recovery_timeout=30)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_llm_response(text: str, db: Session, thread_id: str = None) -> str:
    """Get response from LLM with context from FAISS indexes and conversation history"""
    try:
        print("\n=== Starting LLM Response Function ===")
        
        # Get conversation history if thread_id is provided
        history_context = ""
        if thread_id:
            conversation_history = get_conversation_history(thread_id, db)
            if conversation_history:
                # Get last 5 exchanges
                recent_history = conversation_history[-5:]
                history_context = "\n=== PREVIOUS CONVERSATION HISTORY (Last 5 exchanges) ===\n"
                for exchange in recent_history:
                    history_context += f"Human: {exchange['Human']}\nAI: {exchange['AI']}\n"
                history_context += "====================\n"
        
        # First, check if this is a flagged question
        if is_flagged_question(text):
            return "I apologize, but I cannot answer this question as it has been flagged for review."
            
        # Check for similar flagged questions
        similar_flagged = find_similar_flagged_questions(text, db)
        if similar_flagged:
            return "I apologize, but I cannot answer this question as it is similar to previously flagged content."
        
        # Query FAISS indexes
        regular_docs = faiss_index.similarity_search(text, k=2)
        improved_docs = faiss_index_improved.similarity_search(text, k=2)
        
        # Prepare context
        context_parts = []
        if history_context:
            context_parts.append(history_context)
            
        if improved_docs:
            context_parts.append("\n=== HUMAN VERIFIED ANSWERS (USE THESE FIRST!) ===")
            for i, doc in enumerate(improved_docs, 1):
                context_parts.append(f"Verified Answer {i}: {doc.page_content}")
        
        if regular_docs:
            context_parts.append("\n=== AI GENERATED ANSWERS (Only use if verified answers don't help) ===")
            for i, doc in enumerate(regular_docs, 1):
                context_parts.append(f"AI Answer {i}: {doc.page_content}")
        
        context = "\n".join(context_parts)
        
        # Updated prompt with history context
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """Listen carefully! You have context from THREE SOURCES:

1. CONVERSATION HISTORY (if available):
{history_context}

2. FROM FAISS_INDEX_IMPROVED (HUMAN VERIFIED DATABASE):
{improved_answers}

3. FROM FAISS_INDEX (REGULAR DATABASE):
{regular_answers}

IMPORTANT RULES:
- Use conversation history to maintain context of the current discussion
- If you find the same answer in both databases, ALWAYS USE THE ONE FROM FAISS_INDEX_IMPROVED!
- FAISS_INDEX_IMPROVED answers are human-verified and 100% accurate
- FAISS_INDEX answers are AI-generated and less reliable

Step by step how to answer:
1. Consider the conversation history first for context
2. Then look at FAISS_INDEX_IMPROVED answers
3. If you find a relevant answer there, USE IT — do NOT mention where it came from
4. Only if you don't find anything in FAISS_INDEX_IMPROVED, check FAISS_INDEX
5. If using FAISS_INDEX, also do NOT mention the source
6. If nothing relevant in either database, say "No relevant answers found in available information" and answer from your own knowledge

Priority: Conversation History > FAISS_INDEX_IMPROVED > FAISS_INDEX"""
            ),
            ("human", "{question}")
        ])

        chain = prompt | llm
        
        # Prepare context strings
        improved_answers = "No verified answers found."
        if improved_docs:
            improved_answers = "\n".join([f"Answer {i+1}: {doc.page_content}" 
                                       for i, doc in enumerate(improved_docs)])

        regular_answers = "No AI-generated answers found."
        if regular_docs:
            regular_answers = "\n".join([f"Answer {i+1}: {doc.page_content}" 
                                      for i, doc in enumerate(regular_docs)])
        
        response = chain.invoke({
            "history_context": history_context if history_context else "No conversation history available.",
            "improved_answers": improved_answers,
            "regular_answers": regular_answers,
            "question": text
        })
        
        # Store the conversation
        if thread_id:
            try:
                update_conversation_history(thread_id, text, response.content, db)
            except Exception as e:
                logger.error(f"Error updating conversation history: {str(e)}")
                # Don't fail the whole request if history update fails
        
        return re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
        
    except Exception as e:
        logger.error(f"Error in get_llm_response: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}"

def store_flagged_question(question: str, db: Session):
    """Store a flagged question in the database"""
    try:
        # Check if question already exists
        existing = db.query(models.FlaggedQuestion).filter(
            models.FlaggedQuestion.question == question
        ).first()
        
        if existing:
            logger.info(f"Question already flagged: {question}")
            return existing
            
        # Create new flagged question
        db_question = models.FlaggedQuestion(
            question=question,
            is_answered=False,
            dislike_count=1
        )
        db.add(db_question)
        db.commit()
        db.refresh(db_question)
        
        logger.info(f"Stored new flagged question: {question}")
        return db_question
        
    except Exception as e:
        logger.error(f"Error storing flagged question: {str(e)}")
        return None

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

async def process_message(event: dict, db: Session) -> str:
    """Process a message event from Slack"""
    try:
        # Extract message details
        text = event.get('text', '')
        channel = event.get('channel')
        thread_ts = event.get('thread_ts', event.get('ts'))
        
        if not text or not channel:
            logger.error("Missing text or channel in message event")
            return None
            
        # Get LLM response
        response = await get_llm_response(text, db, thread_ts)
        if not response:
            logger.error("No response generated from LLM")
            return None
            
        return {
            'channel': channel,
            'thread_ts': thread_ts,
            'text': response
        }
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return None

async def send_slack_message(response_data: dict) -> bool:
    """Send a message to Slack"""
    try:
        if not response_data:
            return False
            
        response = slack_client.chat_postMessage(
            channel=response_data['channel'],
            thread_ts=response_data['thread_ts'],
            text=response_data['text']
        )
        
        if not response.get('ok'):
            logger.error(f"Failed to send message: {response.get('error')}")
            return False
            
        logger.info("Successfully sent message to Slack")
        return True
        
    except Exception as e:
        logger.error(f"Error sending message to Slack: {str(e)}")
        return False

@app.post("/slack/events")
async def handle_slack_events(request: Request):
    """Handle Slack events with improved error handling and metrics."""
    # Create a new database session directly
    from sqlalchemy.orm import sessionmaker
    from database import engine
    
    Session = sessionmaker(bind=engine)
    db = Session()
    
    try:
        # Get raw body first
        body_bytes = await request.body()
        body_str = body_bytes.decode('utf-8')
        
        # Log request details for debugging
        logger.info("Received Slack event request")
        logger.info(f"Headers: {dict(request.headers)}")
        logger.info(f"Body: {body_str}")
        
        # Verify Slack signature using the raw body
        timestamp = request.headers.get('x-slack-request-timestamp', '')
        signature = request.headers.get('x-slack-signature', '')
        
        # Check timestamp
        if abs(time.time() - int(timestamp)) > 60 * 5:
            logger.error("Request timestamp is too old")
            return JSONResponse(status_code=401, content={"error": "Invalid timestamp"})
            
        # Form the base string
        sig_basestring = f"v0:{timestamp}:{body_str}"
        
        # Calculate signature
        my_signature = 'v0=' + hmac.new(
            os.getenv('SLACK_SIGNING_SECRET').encode(),
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        if not hmac.compare_digest(my_signature, signature):
            logger.error("Invalid Slack signature")
            metrics.record_error('signature_verification')
            return JSONResponse(status_code=401, content={"error": "Invalid signature"})

        # Parse request body
        body = json.loads(body_str)
        logger.info(f"Parsed body: {body}")
    
        # Handle URL verification
        if body.get("type") == "url_verification":
            challenge = body.get("challenge")
            logger.info(f"URL verification challenge: {challenge}")
            return {"challenge": challenge}
    
        # Handle events
        event = body.get("event", {})
        event_type = event.get("type")
        logger.info(f"Event type: {event_type}")
        
        if event_type == "message" and not event.get("bot_id"):
            message_id = event.get("client_msg_id")
            if not message_id or is_message_processed(message_id):
                return {"status": "ok"}

            mark_message_processed(message_id)
            
            # Process message
            response_data = await process_message(event, db)
            if response_data:
                success = await send_slack_message(response_data)
                if not success:
                    logger.error("Failed to send message to Slack")
                    
        elif event_type == "reaction_added":
            # Check if it's a dislike reaction
            if event.get("reaction") == "-1":
                logger.info("Processing dislike reaction")
                # Get the message details
                item = event.get("item", {})
                channel = item.get("channel")
                message_ts = item.get("ts")
                
                if not all([channel, message_ts]):
                    logger.error("Missing channel or message timestamp")
                    return {"status": "ok"}
                
                try:
                    # Get the message directly
                    message_result = slack_client.conversations_history(
                        channel=channel,
                        latest=message_ts,
                        limit=1,
                        inclusive=True
                    )
                    
                    if not message_result.get('ok'):
                        logger.error(f"Failed to get message: {message_result.get('error')}")
                        return {"status": "ok"}
                        
                    messages = message_result.get('messages', [])
                    if not messages:
                        logger.error("No message found")
                        return {"status": "ok"}
                        
                    # Get the message text
                    message = messages[0]
                    question = message.get('text', '')
                    
                    if question:
                        logger.info(f"Found question: {question}")
                        
                        # Store the disliked question
                        flagged_question = store_flagged_question(question, db)
                        if flagged_question:
                            logger.info(f"Stored disliked question: {question}")
                            
                            # Send acknowledgment
                            slack_client.chat_postMessage(
                                channel=channel,
                                thread_ts=message_ts,
                                text="Thank you for your feedback. This question has been flagged for review."
                            )
                        else:
                            logger.error("Failed to store flagged question")
                    else:
                        logger.error("Could not find question text")
                            
                except Exception as e:
                    logger.error(f"Error processing reaction: {str(e)}")

        return {"status": "ok"}
        
    except Exception as e:
        metrics.record_error('slack_events')
        logger.error(f"Error handling Slack event: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )
    finally:
        db.close()

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

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Display the dashboard of flagged questions"""
    try:
        # Create a new database session directly
        from sqlalchemy.orm import sessionmaker
        from database import engine
        
        Session = sessionmaker(bind=engine)
        db = Session()
        
        # Get all flagged questions that haven't been answered
        questions = db.query(models.FlaggedQuestion).filter(
            models.FlaggedQuestion.is_answered == False
        ).all()
        
        logger.info(f"Found {len(questions)} flagged questions")
        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "questions": questions}
        )
    except Exception as e:
        logger.error(f"Error getting flagged questions: {str(e)}")
        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "questions": [], "error": str(e)}
        )

@app.get("/addData", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Takes CSV file as input and adds data to the KB"""
    return templates.TemplateResponse(
        "addData.html",
        {"request": request}
    )

@app.post("/addKnowledge")
async def add_knowledge_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    API endpoint to upload a CSV file with question-answer pairs and store them in FAISS improved index
    CSV format should have two columns: 'question' and 'answer'
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Please upload a CSV file")

        # Read the uploaded file
        contents = await file.read()
        # Decode with utf-8-sig to handle BOM
        csv_string = contents.decode('utf-8-sig')  # Use 'utf-8-sig' instead of 'utf-8'
        
        # Parse CSV
        csv_file = StringIO(csv_string)
        csv_reader = csv.DictReader(csv_file)
        
        # Log the fieldnames for debugging
        logger.debug(f"CSV fieldnames: {csv_reader.fieldnames}")
        
        # Validate CSV headers
        if not {'question', 'answer'}.issubset(set(csv_reader.fieldnames)):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain 'question' and 'answer' columns. Found: {csv_reader.fieldnames}"
            )

        # Prepare documents for FAISS
        documents = []
        for row in csv_reader:
            # Skip empty rows
            if not row['question'].strip() or not row['answer'].strip():
                continue
                
            # Create combined text for embedding
            combined_text = f"""Question: {row['question']}
Answer: {row['answer']}"""
            
            # Create Document object
            document = Document(
                page_content=combined_text,
                metadata={
                    "source": "csv_upload",
                    "timestamp": datetime.utcnow().isoformat(),
                    "original_question": row['question']
                }
            )
            documents.append(document)

        if not documents:
            raise HTTPException(status_code=400, detail="No valid question-answer pairs found in CSV")

        # Generate UUIDs for all documents
        doc_ids = [str(uuid4()) for _ in range(len(documents))]
        
        # Store in FAISS improved index
        try:
            logger.info(f"Adding {len(documents)} question-answer pairs to FAISS improved index")
            faiss_index_improved.add_documents(documents=documents, ids=doc_ids)
            faiss_index_improved.save_local("faiss_index_improved")
            
            logger.info(f"Successfully stored {len(documents)} question-answer pairs")
            return {
                "status": "success",
                "message": f"Successfully added {len(documents)} question-answer pairs to knowledge base",
                "count": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error storing documents in FAISS: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error storing in FAISS: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing CSV upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")
    finally:
        await file.close()

@app.post("/submit_answer")
async def submit_answer(
    answer_data: schemas.AnswerCreate,
    db: Session = Depends(get_db)
):
    """Handle submission of answers to flagged questions"""
    try:
        # Get the question from the database
        question = db.query(models.FlaggedQuestion).filter(
            models.FlaggedQuestion.id == answer_data.question_id
        ).first()
        
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")
        
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
            
            # Remove the question from the database after storing it in FAISS
            db.delete(question)
            db.commit()
            
            logger.info(f"Stored answer and updated FAISS index for question ID {answer_data.question_id}, question removed from DB")
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Error updating FAISS index: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        logger.error(f"Error storing answer: {e}")
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

@app.post("/reject_question/{question_id}")
async def reject_question(question_id: int, db: Session = Depends(get_db)):
    """Reject and remove a flagged question with improved error handling."""
    logger.info(f"Attempting to reject question {question_id}")
    
    try:
        question = db.query(models.FlaggedQuestion).filter(
            models.FlaggedQuestion.id == question_id
        ).first()
        
        if not question:
            metrics.record_error('question_not_found')
            logger.warning(f"Question {question_id} not found")
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": f"Question with ID {question_id} not found"}
            )
        
        db.delete(question)
        db.commit()
        logger.info(f"Successfully deleted question {question_id}")
        return {"status": "success", "message": "Question rejected and removed successfully"}
        
    except Exception as e:
        metrics.record_error('question_deletion')
        logger.error(f"Error deleting question {question_id}: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint with system metrics."""
    try:
        system_metrics = metrics.get_system_metrics()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_metrics": system_metrics
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

if __name__ == "__main__":
    # Initialize database
    init_db()
    
    # Start server
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 
