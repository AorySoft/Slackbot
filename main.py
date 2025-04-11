import os
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import UploadFile, File, HTTPException
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
        db.rollback()

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
3. If you find a relevant answer there, USE IT and mention "Based on verified answer from FAISS_INDEX_IMPROVED:"
4. Only if you don't find anything in FAISS_INDEX_IMPROVED, check FAISS_INDEX
5. If using FAISS_INDEX, say "Based on AI-generated answer from FAISS_INDEX:"
6. If nothing relevant in either database, say "No relevant answers found in either database" and answer from your knowledge

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
            update_conversation_history(thread_id, text, response.content, db)
        
        return re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
        
    except Exception as e:
        logger.error(f"Error in get_llm_response: {str(e)}")
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
        
        try:
            # Parse and log JSON body
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
                processed_messages[message_id] = True
                
                # Process user message
                if text and user_id and message_id:  # Only process if we have a message ID
                    try:
                        db = next(get_db())
                        thread_ts = event.get('thread_ts', event.get('ts'))  # Use thread_ts if available, else message ts
                        llm_response = await get_llm_response(text, db, thread_ts)
                        
                        # Send response
                        response = slack_client.chat_postMessage(
                            channel=channel_id,
                            thread_ts=thread_ts,
                            text=llm_response
                        )
                        
                        # Add message ID to processed set
                        processed_messages.add(message_id)
                        print(f"✅ Added message {message_id} to processed set")
                        print("✅ Sent response successfully:", response)
                    except Exception as e:
                        print(f"❌ Error sending response: {str(e)}")
                        logger.error(f"Error sending response: {str(e)}", exc_info=True)

            # Handle reaction events (unchanged from original)
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
                                
                                # Generate embedding for the question
                                try:
                                    question_embedding = embeddings.embed_query(user_question)
                                    question_embedding_json = json.dumps(question_embedding)
                                    print("✅ Generated question embedding")
                                except Exception as e:
                                    print(f"❌ Error generating embedding: {str(e)}")
                                    question_embedding_json = None
                                
                                # Store both question and bot's response
                                db_question = models.FlaggedQuestion(
                                    question=user_question,
                                    llm_response=bot_response,
                                    question_embedding=question_embedding_json,
                                    dislike_count=1
                                )
                                db.add(db_question)
                                db.commit()
                                print("✅ Successfully stored disliked Q&A pair with embedding")
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


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Display the dashboard of flagged questions"""
    questions = get_flagged_questions(db)
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "questions": questions}
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