import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    "Global configuration Setting for Hybrid RAG Application."
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    
    
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE",1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 2000))
    
    
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 768))  # Model-dependent
