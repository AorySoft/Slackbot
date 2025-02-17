from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class AnswerCreate(BaseModel):
    """Schema for submitting an answer to a flagged question"""
    question_id: int
    correct_answer: str

class FlaggedQuestion(BaseModel):
    """Schema for returning flagged questions"""
    id: int
    question: str
    llm_response: Optional[str] = None
    correct_answer: Optional[str] = None
    is_answered: bool
    dislike_count: int
    timestamp: datetime
    embedding_id: Optional[str] = None

    class Config:
        from_attributes = True 