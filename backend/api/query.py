from fastapi import APIRouter, HTTPException

from backend.agents import file_agent
from backend.models.chat import (
    AskRequest,
    AskResponse,
    ChatMessageResponse,
    SourceCitationResponse,
)
from backend.services.chat_store import SourceCitation, chat_store

router = APIRouter(tags=["query"])

def is_meaningful_answer(answer: str) -> bool:
    normalized = answer.strip().lower()

    if not normalized:
        return False

    weak_answers = {
        "i don't know",
        "i do not know",
        "unknown",
        "not sure",
        "no answer",
        "none",
    }

    if normalized in weak_answers:
        return False

    if len(normalized) < 12:
        return False

    return True


@router.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
    if file_agent.qa_chain is None:
        raise HTTPException(status_code=400, detail="No file uploaded yet.")

    session = chat_store.get_session(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    messages = chat_store.get_messages(payload.session_id)

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

    raw_answer = result.get("answer", "")
    source_documents = result.get("source_documents", [])

    answer = raw_answer.strip() if isinstance(raw_answer, str) else ""

    if not is_meaningful_answer(answer):
        answer = (
            "I found relevant sections in the document, but I could not generate a confident answer. "
            "Please try a more specific question."
        )
        source_documents = []

    sources: list[SourceCitation] = []

    if is_meaningful_answer(answer):
        for doc in source_documents:
            sources.append(
                SourceCitation(
                    filename=doc.metadata.get("filename"),
                    file_url=doc.metadata.get("file_url"),
                    page=doc.metadata.get("page"),
                    chunk_index=doc.metadata.get("chunk_index"),
                    content=doc.page_content,
                )
            )

    chat_store.add_message(payload.session_id, "user", payload.question)
    chat_store.add_message(
        payload.session_id,
        "assistant",
        answer,
        sources=sources,
    )

    updated_messages = chat_store.get_messages(payload.session_id)

    return AskResponse(
        answer=answer,
        session_id=payload.session_id,
        messages=[
            ChatMessageResponse(
                role=msg.role,
                content=msg.content,
                sources=[
                    SourceCitationResponse(
                        filename=source.filename,
                        file_url=source.file_url,
                        page=source.page,
                        chunk_index=source.chunk_index,
                        content=source.content,
                    )
                    for source in (msg.sources or [])
                ]
                if msg.sources
                else None,
            )
            for msg in updated_messages
        ],
    )