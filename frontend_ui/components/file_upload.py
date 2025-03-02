import streamlit as st
import os

def save_uploaded_file(uploaded_file, save_dir="uploads"):
    """Saves uploaded file to the specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def file_upload_ui():
    """Handles file upload UI in Streamlit."""
    st.sidebar.subheader("📤 Upload PDF Files")
    uploaded_files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        file_paths = [save_uploaded_file(file) for file in uploaded_files]
        st.sidebar.success("✅ Files uploaded successfully!")
        return file_paths
    return []
