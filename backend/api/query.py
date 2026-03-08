from fastapi import APIRouter, HTTPException
from backend.agents.file_agent import qa_agent  # Import the global agent

router = APIRouter()

@router.post("/ask")
async def ask_question(question: str):
    if qa_agent is None:
        raise HTTPException(status_code=400, detail="No file uploaded yet.")
    
    answer = qa_agent.run(question)
    return {"question": question, "answer": answer}