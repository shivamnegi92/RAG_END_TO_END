import streamlit as st
from pinecone_rag import hybrid_search

st.title("📄 AI-Powered Document Chatbot")
st.write("Search through stored PDFs using hybrid retrieval!")

query = st.text_input("Enter your query:")

if query:
    results = hybrid_search(query)
    st.write("### 🔍 Top Matching Documents:")
    for doc in results:
        st.write(f"📌 {doc}")
