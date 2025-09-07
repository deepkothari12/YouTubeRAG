import os
import logging
from langchain_community.vectorstores import Chroma
from scripts.embedding_model import get_embedding_function

# Initialize embedding function once
_embedding_function = get_embedding_function()

def Data_storing(docs, collection_name="youtube_rag", persist_dir=None): ##persist_dir --> parameter tells Chroma where on disk to store the database files.
    """Store documents in Chroma vector DB."""
    persist_dir = persist_dir or os.getenv("PERSIST_DIR", "./data/youtube_rag")
    try:
        db = Chroma.from_documents(
            documents=docs,
            embedding=_embedding_function,
            collection_name=collection_name,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )
        db.persist()
        logging.info(f"Stored {len(docs)} docs in collection '{collection_name}' at {persist_dir}")
        return db
    except Exception as e:
        logging.error(f"Failed to store documents: {e}")
        raise

def load_db(collection_name="youtube_rag", persist_dir=None):
    """Load existing Chroma DB."""
    persist_dir = persist_dir or os.getenv("PERSIST_DIR", "./data/youtube_rag")
    try:
        db = Chroma(
            collection_name=collection_name,
            embedding_function=_embedding_function,
            persist_directory=persist_dir
        )
        logging.info(f"Loaded collection '{collection_name}' from {persist_dir}")
        return db
    except Exception as e:
        logging.error(f"Failed to load database: {e}")
        raise
    
