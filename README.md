##🤖 RAG_AI_Chatbot_Multi_PDF_Support[Streamlit + Groq (LLaMA3)]

This project is a fully interactive **RAG (Retrieval-Augmented Generation) Chatbot** built with **Streamlit** and **LangChain**, powered by **Groq's blazing-fast LLaMA3-8B**. It allows you to upload **multiple PDFs**, ask questions, and get precise, context-aware answers in a conversational format.

---

## 🚀 Features

- 📤 Upload **multiple PDF documents**
- 🤖 LLM-powered answers via **Groq LLaMA3-8B**
- 📑 **Document viewer** in sidebar (first 3 pages per PDF)
- 🔄 **Loading spinner** while processing queries
- 💬 Chat-style history and interface
- 🧠 Uses **HuggingFace sentence-transformers** for embeddings

---

## 📸 Demo

> Upload PDFs → Ask Questions → Get Context-Aware Answers

![Screenshot 2025-06-17 225618](https://github.com/user-attachments/assets/dc83f75a-465b-40b8-bf8a-9f322f3f1d03)

---

## 🧱 Tech Stack

| Layer        | Tool/Library                      |
|--------------|-----------------------------------|
| UI           | [Streamlit](https://streamlit.io) |
| Backend      | [LangChain](https://www.langchain.com) |
| LLM Host     | [Groq](https://console.groq.com/) (LLaMA3) |
| Embeddings   | `sentence-transformers/all-MiniLM-L6-v2` |
| PDF Handling | PyPDF2, LangChain PDF Loader      |

---
---
## 🗂️ Project Structure
```bash
rag_chatbot/
├── app.py               # Streamlit frontend
├── rag_engine.py        # Core RAG logic (PDF loading, LLM response)
├── .env                 # API key for Groq
├── temp/                # Temporary file storage
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation

---



---

## ⚙️ Setup Instructions

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot

### ✅ Step 2: Create Virtual Environment
python -m venv venv
# Activate it
source venv/bin/activate       # On Linux/Mac
venv\Scripts\activate          # On Windows

### ✅ Step 3: Install Dependencies
pip install -r requirements.txt

### requirements.txt
streamlit
python-dotenv
langchain
langchain-community
langchain-core
langchain-groq
PyPDF2
sentence-transformers

### ✅ Step 4: Set Environment Variables
GROQ_API_KEY=your_actual_groq_api_key

### ▶️ Run the Application
streamlit run app.py

## 💡 How It Works

- PDF files are uploaded and stored temporarily in a `/temp/` directory.
- Text is extracted using `PyPDF2` and `LangChain`’s `PyPDFLoader`.
- Text chunks are embedded using `HuggingFace` sentence-transformers (`all-MiniLM-L6-v2`).
- A vectorstore is created and queried via `LangChain's RetrievalQA`.
- User queries are answered using `Groq’s LLaMA3-8B` model, delivering fast and accurate responses grounded in the uploaded content.

---

## ✨ Future Enhancements

- [ ] Source highlighting in answers
- [ ] Export chat to PDF/Markdown
- [ ] Upload `.docx` / `.txt` files
- [ ] Switchable LLMs (OpenAI, Claude, Mixtral)

---

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com)
- [Groq](https://console.groq.com)
- [Streamlit](https://streamlit.io)
- [HuggingFace](https://huggingface.co)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🧠 Author

**Vaibhav Anand**  
Feel free to reach out or contribute!




