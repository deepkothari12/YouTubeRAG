import os
import torch
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cache the embedding model so it's only loaded once not every timee
_embedding_function = None

def get_embedding_function(model_name: str = None):
    global _embedding_function
    if _embedding_function is not None:
        return _embedding_function
    
    try:
        model = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        logging.info(f"Loading embedding model: {model} on {device}")
        _embedding_function = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device}
        )
        return _embedding_function
    
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        raise
