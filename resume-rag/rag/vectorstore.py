import os
import logging
from typing import List
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from rag.embeddings import get_embeddings_model
from rag.parser import extract_text

logger = logging.getLogger("VectorStore")

DEFAULT_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "chroma_db"
) 

def get_vector_store(api_key: str, persist_directory: str = DEFAULT_DB_DIR) -> Chroma:
    """Loads or creates the Chroma vector store."""
    embeddings = get_embeddings_model(api_key)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="resume_rag"
    )

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Splits text into chunks recursively."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

def add_resume_to_vector_store(file_path: str, candidate_id: str, api_key: str, persist_directory: str = DEFAULT_DB_DIR) -> bool:
    """
    Extracts text, chunks it, and adds the chunks with metadata to ChromaDB.
    Clears existing chunks for the same candidate first to avoid duplication.
    """
    try:
        # Extract text
        text = extract_text(file_path)
        if not text.strip():
            logger.warning(f"No text extracted from {file_path}. Skipping vector storage.")
            return False

        # Ensure we delete any existing chunks for this candidate first
        delete_candidate_chunks(candidate_id, api_key, persist_directory)

        # Chunk text
        chunks = chunk_text(text)
        
        # Create langchain documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "candidate_id": candidate_id,
                    "source": os.path.basename(file_path),
                    "chunk_id": f"{candidate_id}_chunk_{i}"
                }
            )
            documents.append(doc)
            
        # Add to Chroma
        db = get_vector_store(api_key, persist_directory)
        db.add_documents(documents)
        logger.info(f"Added {len(documents)} chunks to vector store for candidate {candidate_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error adding resume to vector store: {e}")
        return False

def delete_candidate_chunks(candidate_id: str, api_key: str, persist_directory: str = DEFAULT_DB_DIR) -> bool:
    """Deletes all chunks in ChromaDB associated with the given candidate_id."""
    try:
        db = get_vector_store(api_key, persist_directory)
        # Access the underlying chromadb collection to query/delete
        collection = db._collection
        results = collection.get(where={"candidate_id": candidate_id})
        ids = results.get("ids", [])
        
        if ids:
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} existing chunks for candidate {candidate_id} from vector store.")
        return True
    except Exception as e:
        logger.error(f"Error deleting chunks for candidate {candidate_id}: {e}")
        return False

def clear_vector_store(api_key: str, persist_directory: str = DEFAULT_DB_DIR) -> bool:
    """Deletes all documents inside the vector store."""
    try:
        db = get_vector_store(api_key, persist_directory)
        db.delete_collection()
        logger.info("ChromaDB vector store cleared.")
        return True
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        return False
