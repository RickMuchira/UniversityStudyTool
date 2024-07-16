import os
import pickle
import streamlit as st
import faiss
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv

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

# Function to generate response to user's question
def generate_response(question, vectors):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': question})
    return response['answer']

# Streamlit interface
st.title("University Study Assistant - Ask Questions")

st.subheader("Ask a Question")
question = st.text_input("Enter your question here:")
query_option = st.selectbox("Query type", ("Regular PDFs", "Past Exam Papers"))

if st.button("Get Answer"):
    if query_option == "Regular PDFs":
        index_file = "shared_storage/vectors.index"
        documents_file = "shared_storage/documents.pkl"
    else:
        index_file = "past_paper_vectors.index"
        documents_file = "past_paper_documents.pkl"

    # Load vectors and documents
    index = faiss.read_index(index_file)
    with open(documents_file, "rb") as f:
        documents = pickle.load(f)

    docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
    vectors = FAISS(embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), docstore=docstore, index=index, index_to_docstore_id={i: i for i in range(len(documents))})

    answer = generate_response(question, vectors)
    st.write("Answer:", answer)
