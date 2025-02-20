from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Create SQLAlchemy engine
SQLALCHEMY_DATABASE_URL = "sqlite:///./slack_bot.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

class FlaggedQuestion(Base):
    __tablename__ = "flagged_questions"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    question_embedding = Column(Text, nullable=True)  # To store question embedding as JSON string
    llm_response = Column(Text, nullable=True)
    correct_answer = Column(Text, nullable=True)
    is_answered = Column(Boolean, default=False)
    dislike_count = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    embedding_id = Column(String, nullable=True)  # To store FAISS vector ID
    
    @property
    def combined_text(self):
        """Get combined text for embedding"""
        parts = [self.question]
        if self.correct_answer:
            parts.append(self.correct_answer)
        return " ".join(parts)

# Create all tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 