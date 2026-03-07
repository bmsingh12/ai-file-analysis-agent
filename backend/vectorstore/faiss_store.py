from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_vector_store(docs):

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore