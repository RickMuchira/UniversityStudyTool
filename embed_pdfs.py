# embed_pdfs.py

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import faiss
import pickle

# Load environment variables (if needed)
from dotenv import load_dotenv
load_dotenv()

# Function to process uploaded PDF
def process_uploaded_pdf(uploaded_file):
    # Save the uploaded file to a temporary location
    temp_path = f"./uploads/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Process the PDF (similar to your existing logic)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])
    vectors = FAISS.from_documents(final_documents, embeddings)

    return vectors, final_documents

def main():
    st.title("PDF Uploader and Processor")

    # File upload widget
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)

    if uploaded_file:
        for pdf_file in uploaded_file:
            # Process each uploaded PDF
            vectors, final_documents = process_uploaded_pdf(pdf_file)
            st.success(f"File '{pdf_file.name}' processed successfully!")

if __name__ == "__main__":
    main()
