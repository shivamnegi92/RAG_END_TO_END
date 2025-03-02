import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone client by creating an instance of the Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Define the index specification
index_spec = ServerlessSpec(
    cloud='aws',  # Or your preferred cloud provider
    region='us-east-1'  # Adjust to a region supported by the free plan
)

# Ensure the index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # Adjust this to match your vector dimension
        metric='cosine',  # Or your preferred metric
        spec=index_spec
    )

# Connect to the index
index = pc.Index(INDEX_NAME)

def query_pinecone(query_vector, top_k=5):
    """Search for similar vectors in Pinecone"""
    return index.query(vector=query_vector, top_k=top_k, include_metadata=True)

def upsert_documents(documents, embeddings):
    """Upsert documents into Pinecone"""
    vectors = [(str(i), embeddings[i], {"text": doc}) for i, doc in enumerate(documents)]
    index.upsert(vectors=vectors)

def delete_document(doc_id):
    """Delete a document from Pinecone"""
    index.delete(ids=[str(doc_id)])

def get_index():
    """Return Pinecone index object"""
    return index
