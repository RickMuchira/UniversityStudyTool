import os
import pickle
import hashlib
import streamlit as st
import faiss
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Load environment variables
load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize ChatGroq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Create ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

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

# Function to hash chunks for tracking
def hash_chunk(chunk):
    return hashlib.sha256(chunk.page_content.encode()).hexdigest()

# Function to summarize text using Sumy
def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

# Load previously uploaded files
uploaded_files_history = []
if os.path.exists("shared_storage/uploaded_files_history.pkl"):
    with open("shared_storage/uploaded_files_history.pkl", "rb") as f:
        uploaded_files_history = pickle.load(f)

# Load the questions history
questions_history = {}
if os.path.exists("shared_storage/questions_history.pkl"):
    with open("shared_storage/questions_history.pkl", "rb") as f:
        questions_history = pickle.load(f)

# Streamlit interface
st.title("University Study Assistant")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the section", ["Upload PDFs", "Ask Questions", "Summary"])

if app_mode == "Upload PDFs":
    st.header("PDF Upload and Embedding")

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

        # Save vectors and documents in shared storage
        if not os.path.exists("shared_storage"):
            os.makedirs("shared_storage")

        faiss.write_index(vectors.index, "shared_storage/vectors.index")
        with open("shared_storage/documents.pkl", "wb") as f:
            pickle.dump(final_documents, f)

        uploaded_files_history.extend(new_files)
        with open("shared_storage/uploaded_files_history.pkl", "wb") as f:
            pickle.dump(uploaded_files_history, f)

        st.success("PDFs processed and vectors saved.")

    st.subheader("History of Uploaded PDFs")
    if uploaded_files_history:
        st.write("Uploaded PDFs:")
        for file_name in uploaded_files_history:
            st.write(file_name)
    else:
        st.write("No PDFs uploaded yet.")

elif app_mode == "Ask Questions":
    st.header("Query Interface")

    def generate_response(question, vectors):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': question})
        return response['answer']

    st.subheader("Ask a Question")
    question = st.text_input("Enter your question here:")

    if st.button("Get Answer"):
        # Load vectors and documents from shared storage
        index = faiss.read_index("shared_storage/vectors.index")
        with open("shared_storage/documents.pkl", "rb") as f:
            documents = pickle.load(f)

        docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
        vectors = FAISS(embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), docstore=docstore, index=index, index_to_docstore_id={i: i for i in range(len(documents))})

        answer = generate_response(question, vectors)

        # Track the question
        question_hash = hashlib.sha256(question.encode()).hexdigest()
        if question_hash not in questions_history:
            questions_history[question_hash] = question
            with open("shared_storage/questions_history.pkl", "wb") as f:
                pickle.dump(questions_history, f)

        # Calculate progress
        total_chunks = len(documents)
        queried_chunk_hashes = set(questions_history.keys())
        covered_chunks = len([chunk for chunk in documents if hash_chunk(chunk) in queried_chunk_hashes])
        progress = (covered_chunks / total_chunks)

        st.write("Answer:", answer)
        st.write("Study Progress:", f"{progress * 100:.2f}%")
        st.progress(progress)

elif app_mode == "Summary":
    st.header("Summary of Uploaded PDFs")

    # Load documents from shared storage
    with open("shared_storage/documents.pkl", "rb") as f:
        documents = pickle.load(f)

    pdf_summaries = {}

    for doc in documents:
        source = doc.metadata["source"]
        summary = summarize_text(doc.page_content)
        if source not in pdf_summaries:
            pdf_summaries[source] = summary
        else:
            pdf_summaries[source] += " " + summary

    if pdf_summaries:
        for source, summary in pdf_summaries.items():
            st.subheader(f"Summary of {source}")
            st.write(summary)
    else:
        st.write("No summaries available.")
