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
processed_messages = set()  



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
            
            # Store the response in the database
            db_question = models.FlaggedQuestion(
                question=text,
                llm_response=response.content
            )
            db.add(db_question)
            db.commit()
            
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
    logger.debug("Step 1: Received request to /slack/events")
    
    # Log all headers
    headers = dict(request.headers)
    logger.debug(f"Step 2: Request headers: {headers}")
    
    # Get raw body
    raw_body = await request.body()
    body_str = raw_body.decode()
    logger.debug(f"Step 3: Raw body: {body_str}")
    
    # Parse JSON
    try:
        body = await request.json()
        logger.debug(f"Step 4: Parsed JSON body: {body}")
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return {"error": "Invalid JSON"}

    # Log the type of event
    event_type = body.get("type", "no_type")
    logger.debug(f"Step 5: Event type: {event_type}")
    
    # Handle URL verification
    if event_type == "url_verification":
        challenge = body.get("challenge")
        logger.debug(f"Step 6a: Returning verification challenge: {challenge}")
        return {"challenge": challenge}
    
    # Handle event callbacks
    if event_type == "event_callback":
        event = body.get("event", {})
        logger.debug(f"Step 6b: Got event: {event}")
        
        if event.get("type") == "message":
            channel_id = event.get('channel')
            user_id = event.get('user')
            text = event.get('text', '')
            ts = event.get('ts', '')
            bot_id = event.get('bot_id')
            event_id = body.get('event_id', '')  # Get the event ID
            
            # Debug print for message event
            print(f"\n=== Message Event Debug ===")
            print(f"Event ID: {event_id}")
            print(f"User ID: {user_id}")
            print(f"Bot ID: {BOT_ID}")
            print(f"Event bot_id: {bot_id}")
            print(f"Text: {text}")
            print(f"Timestamp: {ts}")
            print("=========================")
            
            # Check if we've already processed this message
            if event_id in processed_messages:
                print(f"Skipping duplicate message with event_id: {event_id}")
                return {"ok": True}
            
            # Skip bot messages - check both bot_id and user_id
            if bot_id or user_id == BOT_ID:
                print("Skipping bot message")
                return {"ok": True}
            
            # Skip messages without text
            if not text:
                print("Skipping empty message")
                return {"ok": True}
            
            # Add event_id to processed messages
            processed_messages.add(event_id)
            
            # Maintain a reasonable size for processed_messages set
            if len(processed_messages) > 1000:  # Limit the size to prevent memory issues
                processed_messages.clear()  # Clear old messages periodically
            
            logger.debug("Step 7: Processing message event")
            logger.info("----------------------------------------")
            logger.info(f"Message from user: {user_id}")
            logger.info(f"Text: {text}")
            logger.info(f"Channel: {channel_id}")
            logger.info("----------------------------------------")

            # Track message count
            if user_id:
                message_counts[user_id] = message_counts.get(user_id, 0) + 1
                logger.debug(f"Updated message count for user {user_id}: {message_counts[user_id]}")

            # Handle 'start' command
            if text and text.lower() == 'start':
                send_welcome_message(f"@{user_id}", user_id)
                return {"ok": True}
            
            # Get LLM response for all messages
            if text:
                try:
                    db = next(get_db())
                    llm_response = await get_llm_response(text, db)
                    slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=event.get('ts'),
                        text=llm_response
                    )
                    logger.info("Sent LLM response")
                except Exception as e:
                    logger.error(f"Failed to send LLM response: {e}")

        # Handle reaction events
        elif event.get("type") == "reaction_added":
            reaction = event.get('reaction', '')
            item = event.get('item', {})
            
            # Handle thumbs down reaction
            if reaction == 'thumbsdown':
                try:
                    # Get the original message
                    result = slack_client.conversations_history(
                        channel=item.get('channel'),
                        latest=item.get('ts'),
                        limit=1,
                        inclusive=True
                    )
                    
                    if result['messages']:
                        original_message = result['messages'][0]['text']
                        db = next(get_db())
                        store_flagged_question(original_message, db)
                        
                        # Notify in thread that the feedback was recorded
                        slack_client.chat_postMessage(
                            channel=item.get('channel'),
                            thread_ts=item.get('ts'),
                            text="Thank you for the feedback. This question has been flagged for review."
                        )
                except Exception as e:
                    logger.error(f"Error handling thumbsdown reaction: {e}")
    
    return {"ok": True}

@app.get("/message-count/{user_id}")
async def get_message_count(user_id: str):
    """Get message count for a specific user"""
    count = message_counts.get(user_id, 0)
    return {"user_id": user_id, "message_count": count}

@app.get("/test_message")
async def test_message():
    """Test sending a message to verify Slack token"""
    try:
        channel_id = os.getenv("SLACK_CHANNEL_ID")
        response = slack_client.chat_postMessage(
            channel=channel_id,
            text="Test message from bot!"
        )
        logger.info(f"Test message response: {response}")
        
        # Return simplified response
        return {
            "status": "success",
            "message_sent": True,
            "channel": response.get("channel"),
            "timestamp": response.get("ts")
        }
    except Exception as e:
        logger.error(f"Error sending test message: {e}")
        return {
            "status": "error",
            "message_sent": False,
            "error": str(e)
        }

@app.post("/test_webhook")
async def test_webhook(request: Request):
    """Test endpoint to verify webhook is accessible"""
    body = await request.body()
    headers = dict(request.headers)
    logger.info("----------------------------------------")
    logger.info("TEST WEBHOOK CALLED")
    logger.info(f"Headers: {headers}")
    logger.info(f"Body: {body.decode()}")
    logger.info("----------------------------------------")
    return {"status": "received"}

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