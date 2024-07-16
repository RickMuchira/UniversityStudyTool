from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import pickle
import faiss

# Load environment variables
load_dotenv()

# Load the GROQ and OpenAI API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Flask app
app = Flask(__name__)

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

# Load vectors and documents
def load_faiss_data():
    index = faiss.read_index("shared_storage/vectors.index")
    with open("shared_storage/documents.pkl", "rb") as f:
        documents = pickle.load(f)
    return index, documents

index, documents = load_faiss_data()

# Create the embeddings and docstore objects
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})

# Create the FAISS vector store
vectors = FAISS(embedding_function=embedding_function, docstore=docstore, index=index, index_to_docstore_id={i: i for i in range(len(documents))})

# Handle incoming messages from Twilio webhook
@app.route('/webhook', methods=['POST'])
def webhook():
    incoming_msg = request.values.get('Body', '').lower()
    response = generate_response(incoming_msg)
    twilio_response = MessagingResponse()
    twilio_response.message(response)
    return str(twilio_response)

# Generate response to user's question
def generate_response(question):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': question})
    return response['answer']

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
