from scripts.indexing import indexing_chunks
from rag.vector_store import Data_storing, load_db
from dotenv import load_dotenv
import logging
import os
from openai import OpenAI

video_id = "5AfJ0N3MvpA" 

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError(f"No API Key for Hugging Face  ")


##configrationsssss
MODEL_NAME = os.getenv("MODEL_NAME")
RETRIEVER_K = int(os.getenv("RETRIEVER_K" , 4))

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



# Indexingggg 
chunks = indexing_chunks(video_id = video_id, languages=['en'])

# Storingggggggg --> Vector DataBase
db_store = Data_storing(docs=chunks)


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

##load the Databse to used for      
db = load_db()
retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})


def ask_question(query: str):
    
    docs = retriever.invoke(query)
    # print("--> docs" , docs)
    # Combine retrieved chunks into a single context string
    context = "\n\n".join([doc.page_content for doc in docs])
    # print(context)
    # Build messages for your custom function
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers using the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    # Call your streaming generator
    # print("Answer:\n")\
    logging.info("Answer:")
    for chunk in gpt_models_streaming(messages):
        print(chunk , end="", flush=True)


ask_question("Can you tell me about RAG from give contextvin just 5 lines")
    