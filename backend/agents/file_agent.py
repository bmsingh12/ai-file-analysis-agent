from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# Global variable to store the agent
qa_agent = None

def init_agent(docs):
    """Initialize the RetrievalQA agent from documents."""
    global qa_agent

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    qa_agent = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa_agent