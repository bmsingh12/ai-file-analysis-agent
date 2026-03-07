from fastapi import APIRouter

router = APIRouter()

@router.post("/ask")
async def ask_question(question: str):
    answer = agent.run(question)
    return {"answer": answer}