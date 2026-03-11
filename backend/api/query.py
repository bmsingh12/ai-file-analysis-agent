from fastapi import APIRouter, HTTPException

from backend.agents import file_agent
from backend.models.chat import AskRequest, AskResponse, ChatMessageResponse
from backend.services.chat_store import chat_store

router = APIRouter(tags=["query"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
    if file_agent.qa_chain is None:
        raise HTTPException(status_code=400, detail="No file uploaded yet.")

    session = chat_store.get_session(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    messages = chat_store.get_messages(payload.session_id)

    # Build LangChain-compatible history: [(human, ai), ...]
    chat_history: list[tuple[str, str]] = []
    current_human: str | None = None

    for msg in messages:
        if msg.role == "user":
            current_human = msg.content
        elif msg.role == "assistant" and current_human is not None:
            chat_history.append((current_human, msg.content))
            current_human = None

    try:
        result = file_agent.ask_with_history(
            question=payload.question,
            chat_history=chat_history,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = result["answer"]

    chat_store.add_message(payload.session_id, "user", payload.question)
    chat_store.add_message(payload.session_id, "assistant", answer)

    updated_messages = chat_store.get_messages(payload.session_id)

    return AskResponse(
        answer=answer,
        session_id=payload.session_id,
        messages=[
            ChatMessageResponse(role=msg.role, content=msg.content)
            for msg in updated_messages
        ],
    )