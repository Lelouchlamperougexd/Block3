import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vector store
vector_store = None

def process_document(file):
    """Process uploaded document and store in vector store"""
    global vector_store
    
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    
    # Load and split the document
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(pages)
    
    # Create or update vector store
    if vector_store is None:
        vector_store = FAISS.from_documents(splits, embeddings)
    else:
        vector_store.add_documents(splits)
    
    # Clean up
    os.remove("temp.pdf")
    return len(splits)

def main():
    st.title("Document Processing with RAG")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                num_chunks = process_document(uploaded_file)
                st.success(f"Document processed successfully! Created {num_chunks} chunks.")
    
    # Query interface
    query = st.text_input("Enter your query:")
    if query and vector_store is not None:
        # Search for similar documents
        docs = vector_store.similarity_search(query, k=3)
        
        st.subheader("Search Results")
        for i, doc in enumerate(docs):
            st.write(f"Result {i+1}:")
            st.write(f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 0)}")
            st.write(doc.page_content)
            st.write("---")

if __name__ == "__main__":
    main() 