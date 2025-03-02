import streamlit as st
import os
import time
from sentence_transformers import SentenceTransformer
from archive.pinecone_rag import query_pinecone, upsert_documents, get_index
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load an embedding model
embed_model = SentenceTransformer("paraphrase-mpnet-base-v2")  # 1536-dimensional embeddings

# Streamlit UI Configuration
st.set_page_config(page_title="AI Document Chatbot", layout="wide")
st.sidebar.title("⚙️ Configuration")

# Response Mode Selection
RESPONSE_MODES = {
    "Concise": "Provide a brief, to-the-point answer.",
    "Detailed": "Offer a comprehensive, in-depth explanation.",
    "ELI5": "Explain the answer as if to a 5-year-old.",
    "Research Aligned": "Present a well-structured, research-oriented response."
}
response_mode = st.sidebar.radio("Select Response Mode", list(RESPONSE_MODES.keys()))

# UI Styling
st.title("📘 AI-Powered Document Chatbot")
st.markdown("### Upload a PDF and chat with your documents using AI!")

# File Upload Handler
def save_uploaded_file(uploaded_file):
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

# Function to Load and Process PDF
def load_and_process_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(documents)

# Function to Perform Hybrid Search and AI Response Generation
def hybrid_search_and_generate_answer(user_query, document_chunks):
    # Convert user query to vector
    query_vector = embed_model.encode(user_query).tolist()

    # Query Pinecone for relevant documents
    results = query_pinecone(query_vector)

    # Get the most relevant documents
    relevant_docs = [match['metadata']['text'] for match in results['matches']]

    # Construct context for AI response generation
    context_text = "\n\n".join(relevant_docs)
    
    # Use AI to generate a response (use appropriate AI logic)
    ai_response = f"Answer generated using {RESPONSE_MODES[response_mode]}: \n{context_text}"
    
    return ai_response

# ---------- Streamlit UI ----------
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    st.info("📄 Document uploaded successfully. Processing...")

    with st.spinner("🔍 Indexing document..."):
        document_chunks = load_and_process_pdf(saved_path)
        upsert_documents([doc.page_content for doc in document_chunks], [embed_model.encode(doc.page_content) for doc in document_chunks])
    st.success("✅ Document processed! Ask your question below.")

    user_input = st.text_input("Enter your question about the document...")

    if user_input:
        with st.spinner("🤖 Thinking..."):
            ai_response = hybrid_search_and_generate_answer(user_input, document_chunks)
        st.write(ai_response)
