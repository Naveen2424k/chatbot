import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger("Embeddings")

def get_embeddings_model(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Generative AI Embeddings model.
    Uses models/text-embedding-004 by default.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing GoogleGenerativeAIEmbeddings: {e}")
        raise e
