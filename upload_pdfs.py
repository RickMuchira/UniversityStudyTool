import os
import pickle
import streamlit as st
import faiss
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Function to perform vector embedding
def embed_pdfs(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = []

    for doc in docs:
        chunks = text_splitter.split_documents([doc])
        final_documents.extend(chunks)

    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors, final_documents

# Load previously uploaded files
uploaded_files_history = []
if os.path.exists("uploaded_files_history.pkl"):
    with open("uploaded_files_history.pkl", "rb") as f:
        uploaded_files_history = pickle.load(f)

past_paper_files_history = []
if os.path.exists("past_paper_files_history.pkl"):
    with open("past_paper_files_history.pkl", "rb") as f:
        past_paper_files_history = pickle.load(f)

# Streamlit interface
st.title("University Study Assistant - Upload PDFs")

option = st.selectbox("Choose the type of PDF to upload", ("Regular PDFs", "Past Exam Papers"))

uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.write("Processing PDFs...")
    all_docs = []
    new_files = []

    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        doc = Document(page_content=text, metadata={"source": uploaded_file.name})
        all_docs.append(doc)
        new_files.append(uploaded_file.name)

    vectors, final_documents = embed_pdfs(all_docs)

    if option == "Regular PDFs":
        faiss.write_index(vectors.index, "vectors.index")
        with open("shared_storage/documents.pkl", "wb") as f:
            pickle.dump(final_documents, f)

        uploaded_files_history.extend(new_files)
        with open("uploaded_files_history.pkl", "wb") as f:
            pickle.dump(uploaded_files_history, f)
    else:
        faiss.write_index(vectors.index, "past_paper_vectors.index")
        with open("past_paper_documents.pkl", "wb") as f:
            pickle.dump(final_documents, f)

        past_paper_files_history.extend(new_files)
        with open("past_paper_files_history.pkl", "wb") as f:
            pickle.dump(past_paper_files_history, f)

    st.success("PDFs processed and vectors saved.")

st.subheader("History of Uploaded PDFs")
if uploaded_files_history:
    st.write("Regular PDFs:")
    for file_name in uploaded_files_history:
        st.write(file_name)
else:
    st.write("No regular PDFs uploaded yet.")

if past_paper_files_history:
    st.write("Past Exam Papers:")
    for file_name in past_paper_files_history:
        st.write(file_name)
else:
    st.write("No past exam papers uploaded yet.")
