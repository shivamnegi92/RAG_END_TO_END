import streamlit as st

def display_response(response):
    """Displays the AI-generated response in Streamlit."""
    st.subheader("🤖 AI Response")
    st.markdown(response)
