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

class ConversationHistoryBase(BaseModel):
    thread_id: str
    conversation: str  # JSON string of list[dict]

class ConversationHistoryCreate(ConversationHistoryBase):
    pass

class ConversationHistory(ConversationHistoryBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True





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


