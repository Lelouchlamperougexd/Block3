import streamlit as st
from constitution_loader import load_documents
from qa_engine import init_vectorstore, ask_question

st.set_page_config(page_title="Kazakhstan Constitution Assistant")
st.title("🇰🇿 AI Assistant - Constitution of Kazakhstan")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_files = st.file_uploader("📄 Upload documents", accept_multiple_files=True, type=["pdf", "txt"])

if uploaded_files:
    with st.spinner("🔄 Processing documents..."):
        documents = load_documents(uploaded_files)
        st.session_state.vectorstore = init_vectorstore(documents)
    st.success("✅ Documents indexed!")

question = st.text_input("❓ Ask a question about the Constitution:")

if st.button("Ask") and question:
    if st.session_state.vectorstore:
        with st.spinner("💬 Generating answer..."):
            answer = ask_question(question, st.session_state.vectorstore)
        st.write("**Answer:**", answer)
    else:
        st.warning("⚠️ Please upload and index documents first.")
