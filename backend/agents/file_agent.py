from __future__ import annotations

import json
import os
from typing import Any, Generator

import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

vector_store: FAISS | None = None
retriever = None

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")


def init_agent(chunks: list[Any]) -> None:
    global vector_store, retriever

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})


def retrieve_documents(question: str):
    if retriever is None:
        raise ValueError("Retriever is not initialized")

    return retriever.invoke(question)


def format_chat_history(chat_history: list[tuple[str, str]]) -> str:
    if not chat_history:
        return "No prior conversation."

    formatted_turns: list[str] = []
    for user_msg, assistant_msg in chat_history:
        formatted_turns.append(f"User: {user_msg}")
        formatted_turns.append(f"Assistant: {assistant_msg}")

    return "\n".join(formatted_turns)


def build_prompt(
    question: str,
    chat_history: list[tuple[str, str]],
    source_documents: list[Any],
) -> str:
    context = "\n\n".join(doc.page_content for doc in source_documents)
    history = format_chat_history(chat_history)

    return f"""
You are a helpful AI assistant answering questions about an uploaded document.

Rules:
- Use only the provided document context.
- If the context is insufficient, clearly say you cannot answer from the document.
- Be concise but useful.
- Do not invent facts.
- Prefer direct answers over vague summaries.

Conversation history:
{history}

Document context:
{context}

User question:
{question}

Answer:
""".strip()


def generate_ollama_answer(prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


def stream_ollama_answer(prompt: str) -> Generator[str, None, None]:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
    }

    with requests.post(url, json=payload, stream=True, timeout=300) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            data = json.loads(line)
            token = data.get("response", "")

            if token:
                yield token

            if data.get("done"):
                break


def ask_with_history(question: str, chat_history: list[tuple[str, str]]) -> dict[str, Any]:
    source_documents = retrieve_documents(question)
    prompt = build_prompt(question, chat_history, source_documents)
    answer = generate_ollama_answer(prompt)

    return {
        "answer": answer,
        "source_documents": source_documents,
    }