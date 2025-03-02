import streamlit as st

def sidebar_configuration():
    """Handles sidebar configuration settings in Streamlit."""
    st.sidebar.title("⚙️ Configuration")
    
    response_modes = {
        "Concise": "Provide a brief, to-the-point answer.",
        "Detailed": "Offer a comprehensive, in-depth explanation.",
        "ELI5": "Explain the answer as if to a 5-year-old.",
        "Research Aligned": "Present a well-structured, research-oriented response."
    }
    response_mode = st.sidebar.radio("Select Response Mode", list(response_modes.keys()))
    
    return response_mode
