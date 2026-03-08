from fastapi import APIRouter, HTTPException
from backend.agents import file_agent

router = APIRouter()

@router.post("/ask")
async def ask_question(question: str):

    if file_agent.qa_agent is None:
        raise HTTPException(status_code=400, detail="No file uploaded yet.")

    answer = file_agent.qa_agent.run(question)

    return {"answer": answer}