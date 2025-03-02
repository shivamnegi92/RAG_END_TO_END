import pinecone
from sentence_transformers import SentenceTransformer
from config import Config

class Retriever:
    """Handles retrieval of relevant document chunks from Pinecone."""

    def __init__(self, embedding_model_name=Config.EMBEDDING_MODEL_NAME):
        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(self.embedding_model_name)
        pinecone.init(api_key=Config.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)
        self.index = pinecone.Index(Config.PINECONE_INDEX_NAME)
    
    def query_pinecone(self, query, top_k=5):
        """Queries Pinecone index with vector search."""
        query_vector = self.model.encode(query).tolist()
        results = self.index.query(query_vector, top_k=top_k, include_metadata=True)
        return [match['metadata']['text'] for match in results['matches'] if 'metadata' in match]
    
    def hybrid_search(self, query, top_k=5):
        """Performs a hybrid search (vector similarity + keyword filtering)."""
        vector_results = self.query_pinecone(query, top_k)
        # Placeholder for keyword search logic (if needed in the future)
        return vector_results
