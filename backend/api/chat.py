from fastapi import APIRouter
from backend.models.chat import CreateSessionResponse
from backend.services.chat_store import chat_store

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/session", response_model=CreateSessionResponse)
async def create_session() -> CreateSessionResponse:
    session = chat_store.create_session()
    return CreateSessionResponse(session_id=session.session_id)