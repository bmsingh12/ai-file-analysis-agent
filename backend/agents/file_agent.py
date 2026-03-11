from __future__ import annotations

from typing import Any

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

qa_chain: ConversationalRetrievalChain | None = None
vector_store: FAISS | None = None


def init_agent(chunks: list[Any]) -> None:
    global qa_chain, vector_store

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(
        model="phi",
        base_url="http://host.docker.internal:11434"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )


def ask_with_history(question: str, chat_history: list[tuple[str, str]]) -> dict[str, Any]:
    if qa_chain is None:
        raise ValueError("QA chain is not initialized")

    return qa_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history,
        }
    )