import streamlit as st

def chat_input_ui():
    """Handles chat input UI in Streamlit."""
    st.subheader("💬 Chat with Your Documents")
    user_query = st.text_input("Enter your question about the document:", "")
    return user_query
