from langchain_community.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def build_agent(vectorstore):

    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        temperature=0
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return chain