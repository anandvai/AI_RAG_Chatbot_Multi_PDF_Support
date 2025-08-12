import streamlit as st
from dotenv import load_dotenv
from rag_engine import get_vectorstore, get_llm_response, extract_pdf_texts

load_dotenv()

st.set_page_config(page_title="📄 RAG Chatbot", layout="wide")
st.title("📄🔍 RAG Chatbot with Multi-PDF Support")

# Sidebar PDF upload
with st.sidebar:
    uploaded_pdfs = st.file_uploader("📤 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_pdfs:
        st.subheader("📚 Preview PDF Content")
        for i, text in enumerate(extract_pdf_texts(uploaded_pdfs), start=1):
            with st.expander(f"📄 Document {i} Preview"):
                st.text(text[:2000])  # Show first 2000 characters

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# User prompt
prompt = st.chat_input("💬 Ask something based on the uploaded PDFs...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not uploaded_pdfs:
        st.warning("⚠️ Please upload at least one PDF file first.")
    else:
        with st.spinner("🔍 Processing and fetching response..."):
            try:
                response = get_llm_response(prompt, uploaded_pdfs)
                st.chat_message("assistant").markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
