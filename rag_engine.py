import os
import uuid
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def save_uploaded_files(uploaded_files):
    file_paths = []
    for file in uploaded_files:
        file_id = str(uuid.uuid4())[:8]
        path = os.path.join(TEMP_DIR, f"{file_id}_{file.name}")
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(path)
    return file_paths

def get_vectorstore(uploaded_files):
    file_paths = save_uploaded_files(uploaded_files)
    loaders = [PyPDFLoader(path) for path in file_paths]

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)

    return index.vectorstore

def get_llm_response(user_query, uploaded_files):
    vectorstore = get_vectorstore(uploaded_files)

    model = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )

    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    result = chain({"query": user_query})
    return result["result"]

def extract_pdf_texts(uploaded_files):
    previews = []
    for file in uploaded_files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages[:3]:  # limit to first 3 pages
            text += page.extract_text() or ""
        previews.append(text)
    return previews
