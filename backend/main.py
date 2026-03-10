from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import upload, query

app = FastAPI()

# Allow your frontend origin
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(query.router)

@app.get("/")
def root():
    return {"message": "AI File Analysis Agent running"}