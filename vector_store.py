import pinecone
from sentence_transformers import SentenceTransformer
from config import Config

class VectorStore:
    """Handles embedding generation and storage in Pinecone."""

    def __init__(self, embedding_model_name=Config.EMBEDDING_MODEL_NAME):
        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(self.embedding_model_name)
        pinecone.init(api_key=Config.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)
        self.index = pinecone.Index(Config.PINECONE_INDEX_NAME)
    
    def embed_text(self, text):
        """Generates an embedding for the given text."""
        return self.model.encode(text).tolist()
    
    def upsert_documents(self, doc_texts):
        """Embeds and upserts document chunks into Pinecone."""
        vectors = [(str(i), self.embed_text(text), {}) for i, text in enumerate(doc_texts)]
        self.index.upsert(vectors)