from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class FlaggedQuestionBase(BaseModel):
    question: str

class FlaggedQuestionCreate(FlaggedQuestionBase):
    llm_response: Optional[str] = None

class AnswerCreate(BaseModel):
    question_id: int
    correct_answer: str

class FlaggedQuestion(FlaggedQuestionBase):
    id: int
    llm_response: Optional[str] = None
    correct_answer: Optional[str] = None
    is_answered: bool
    dislike_count: int
    timestamp: datetime
    embedding_id: Optional[str] = None

    class Config:
        from_attributes = True 