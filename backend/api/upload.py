from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from backend.ingestion.file_loader import load_pdf
from backend.services.chunker import chunk_documents
from backend.agents.file_agent import init_agent

router = APIRouter()

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    saved_path = UPLOAD_DIR / file.filename
    content = await file.read()

    with open(saved_path, "wb") as f:
        f.write(content)

    docs = load_pdf(str(saved_path))
    chunks = chunk_documents(docs)

    for index, chunk in enumerate(chunks):
        chunk.metadata["filename"] = file.filename
        chunk.metadata["file_url"] = f"/uploaded-files/{file.filename}"
        chunk.metadata["chunk_index"] = index

        # Normalize page indexing for UI
        page = chunk.metadata.get("page")
        if isinstance(page, int):
            chunk.metadata["page"] = page + 1

    init_agent(chunks)

    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "file_url": f"/uploaded-files/{file.filename}",
    }