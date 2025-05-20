import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(files):
    documents = []

    for file in files:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if file.name.endswith(".pdf") else ".txt") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
        
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)
