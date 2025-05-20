from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def init_vectorstore(documents):
    embedding = OllamaEmbeddings(model="llama3")
    return Chroma.from_documents(documents, embedding)

def ask_question(query, vectorstore):
    llm = Ollama(model="llama3")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa.run(query)
