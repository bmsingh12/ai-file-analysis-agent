from fastapi import APIRouter, UploadFile, File
from backend.ingestion.file_loader import load_pdf
from backend.services.chunker import chunk_documents
from backend.agents.file_agent import init_agent

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    filename = f"temp_{file.filename}"
    with open(filename, "wb") as f:
        f.write(content)

    docs = load_pdf(filename)
    chunks = chunk_documents(docs)

    # Initialize agent
    init_agent(chunks)

    return {"filename": file.filename, "chunks_created": len(chunks)}