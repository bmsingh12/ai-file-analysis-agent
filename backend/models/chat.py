from pydantic import BaseModel


class CreateSessionResponse(BaseModel):
    session_id: str


class AskRequest(BaseModel):
    session_id: str
    question: str


class SourceCitationResponse(BaseModel):
    filename: str | None = None
    page: int | None = None
    chunk_index: int | None = None
    content: str


class ChatMessageResponse(BaseModel):
    role: str
    content: str
    sources: list[SourceCitationResponse] | None = None


class AskResponse(BaseModel):
    answer: str
    session_id: str
    messages: list[ChatMessageResponse]