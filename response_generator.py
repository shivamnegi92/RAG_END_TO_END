from transformers import pipeline
from config import Config

class ResponseGenerator:
    """Generates AI-powered responses based on retrieved document context."""
    
    def __init__(self, model_name=Config.RESPONSE_MODEL_NAME):
        self.model_name = model_name
        self.generator = pipeline("text-generation", model=self.model_name)
    
    def generate_response(self, query, context, max_length=512):
        """Generates a response using the AI model given user query and document context."""
        prompt = f"Context: {context}\n\nUser Query: {query}\n\nAnswer:"
        response = self.generator(prompt, max_length=max_length, num_return_sequences=1)
        return response[0]['generated_text'].strip()
