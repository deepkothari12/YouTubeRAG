from fastapi import FastAPI, Query
from scripts.indexing import indexing_chunks
from rag.vector_store import Data_storing, load_db
from scripts.embedding_model import get_embedding_function
import logging
import os 
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError(f"No API Key for Hugging Face  ")
MODEL_NAME = os.getenv("MODEL_NAME")
RETRIEVER_K = int(os.getenv("RETRIEVER_K" , 4))

app = FastAPI(title="YouTube RAG API", version="1.0")

### setup login 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                      # Show logs in terminal
        logging.FileHandler("errors.log", mode="a")   # Save errors to file
    ],
    force=True
)


##Model to generate the ANS 
def gpt_models_streaming(messages_list, prompt=None):
    """Generator function for streaming responses"""
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
    
    api_messages = []
    
    if prompt:
        api_messages.append({"role": "system", "content": prompt})
    
    api_messages.extend(messages_list)
    
    try:
        completion = client.chat.completions.create(
            model = MODEL_NAME,
            messages = api_messages,
            stream = True,
        )
        
        # Collect all chunks for the complete response
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content  # Yield each chunk for real-time display
        
        return full_response
    except Exception as e :
        logging.error(f"LLM Request Failes {e}")


# --- Endpoints ---

@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube RAG API is running"}

@app.post("/index/{video_id}")
def index_video(video_id: str, languages: list[str] = Query(["en"])):
    """Fetch transcript, chunk it, and store in vector DB"""
    try:
        chunks = indexing_chunks(video_id, languages=languages)
        db = Data_storing(chunks, collection_name=video_id)
        return {"video_id": video_id, "chunks_stored": len(chunks)}
    except Exception as e:
        logging.error(f"Indexing failed for {video_id}: {e}")
        return {"error": str(e)}

@app.get("/ask/{video_id}")
def ask_question(video_id: str, query: str):
    """Ask a question grounded in a specific video's transcript"""
    try:
        db = load_db(collection_name=video_id)
        retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers using the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        answer = "".join(gpt_models_streaming(messages))
        return {
            "query": query,
            "context_used": context[:500] + "...",
            "answer": answer
        }
    except Exception as e:
        logging.error(f"Query failed for {video_id}: {e}")
        return {"error": str(e)}
