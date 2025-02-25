from dotenv import load_dotenv
import os
import pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
