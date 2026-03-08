from langchain_classic.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

qa_agent = None

def init_agent(chunks):
    global qa_agent

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(
        model="phi",
        base_url="http://host.docker.internal:11434"
    )

    qa_agent = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    print("Agent initialized")