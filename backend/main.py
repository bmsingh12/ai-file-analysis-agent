from fastapi import FastAPI
from backend.api import upload, query

app = FastAPI()

app.include_router(upload.router)
app.include_router(query.router)