from pydantic import BaseModel


class CreateSessionResponse(BaseModel):
    session_id: str


class AskRequest(BaseModel):
    session_id: str
    question: str


class ChatMessageResponse(BaseModel):
    role: str
    content: str


class AskResponse(BaseModel):
    answer: str
    session_id: str
    messages: list[ChatMessageResponse]