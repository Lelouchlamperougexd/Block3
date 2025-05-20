from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

PERSIST_DIR = "chroma_db"
EMBED_MODEL = "llama3"

def init_vectorstore(documents):
    embedding = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents,
        embedding,
        persist_directory=PERSIST_DIR
    )
    vectorstore.persist()
    return vectorstore

def load_vectorstore():
    embedding = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        embedding_function=embedding,
        persist_directory=PERSIST_DIR
    )

def ask_question(query, vectorstore):
    llm = Ollama(model=EMBED_MODEL)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa.run(query)
