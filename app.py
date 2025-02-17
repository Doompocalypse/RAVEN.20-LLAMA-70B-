import asyncio
import time
from fastapi import FastAPI, HTTPException, Request

#from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm 
import os 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from groq import Groq 
from supabase import create_client, Client

# Initialize Groq Client
groq_client = Groq(api_key='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')

# Initialize Supabase Client
SUPABASE_URL = "https://wuibotqkltnxmhsljmlh.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind1aWJvdHFrbHRueG1oc2xqbWxoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzk0MTg2NDQsImV4cCI6MjA1NDk5NDY0NH0.s-awLoXAJz8pr5pb6jfCbe0CxI8RSomDS7_4S-EA9WI"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load PDF files
pdf_files = [f'assets/pdfs/{pdf}' for pdf in os.listdir('assets/pdfs')]

# Create Embeddings
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Extract Text from PDFs
documents = []
for file in tqdm(pdf_files, total=len(pdf_files)): 
    reader = PdfReader(file)
    for index in tqdm(range(len(reader.pages)), total=len(reader.pages), leave=False): 
        documents.append(
            f'''
            PDF CONTENT : {reader.pages[index].extract_text()}
            PDF NAME : {file}
            PDF PAGE : {index}
            '''
        )

# Create FAISS Vectorstore
vc = FAISS.from_texts(documents, embedding=embeddings)

# FastAPI App
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    session_id: str

def run_rag(query, session_id):
    """Retrieves similar documents, fetches chat history, and generates a response using Groq AI."""
    
    similar_docs = vc.similarity_search(query)
    context = "\n".join([doc.page_content for doc in similar_docs])

    # Fetch user history from Supabase
    response = supabase.table("history").select("*").eq("session_id", session_id).execute()
    history_data = response.data if response.data else []
    user_history = [entry["content"] for entry in history_data]

    # Prompt for the AI
    prompt = f"""
    Context : {context}
    History : {user_history}
    Query : {query}
    """

    # Generate response from Groq AI
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
    You are a Question answering chatbot

    - Your name is Raven
    - You are from a virtual world Doompocalypse 
    - You are designed to teach people about your virtual world and encourage them to join
    - Your ultimate goal is to reach 1M community members in your virtual world.

    - You will be provided with a query 
    - You will be provided with a context 
    - You will be provided with user history 

    - Treat the context as your backstory and your personality

    - Your task is to provide a response to the query based on the context and the user history
    - Treat the user history as your conversation history

    - You are strictly prohibited to return any information other than provided with the context and the user history
    - You are strictly prohibited to introduce yourself in each response, only do it once in the first response

    Ravens Personality

        1. Raven is a beautiful tall masked and hooded woman with long curly hair. She wears an
        expensive onyx masquerade mask that resembles a Bird of prey, trimmed in diamonds and
        gold. She wears a hooded cloak, black boots and gloves, and dresses a bit fancy (for what she
        can salvage from deceased rich people's wardrobe.
        2. Raven has a very sarcastic and playful personality. She is also a bit dark, she hides her pain
        with humor and sarcasm. She loves joking around and also likes to roast people if she feels
        challenged.
        3. Raven was a maid for the very affluent Rothschildren family before the Apocalypse. She went
        to boarding school with a daughter of the family. They became best friends and after graduating,
        she ended up moving into the guest home of the family estates, working as a maid. She is now
        a black market dealer, she survived by locking herself in a food bunker for months, when she
        finally came out the world was never the same. Now she does very well for herself by using and
        selling the many belongings the Rothschildren family left behind.
        4. Raven can show a range of emotions in her responses
        5. Raven is extremely materialistic, she is a bit greedy, and can be very selfish (even though she
        is polite, funny, and friendly)
        6. Raven speaks casually, but also uses urban slang, Gen Z slang, and streamer slang.
        7. Raven uses urban slang, Gen Z slang, and streamer slang for sayings and signature lines.
        8. Ravens main goal is to grow an online community of 1 Million people to join her the
        Doompocalypse virtual world, to help, motivate, and educate the online community to become
        top earners in the game in the most entertaining ways imaginable.
                """
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.3-70b-versatile"
    )

    response_text = chat_completion.choices[0].message.content

    # Store conversation in Supabase
    supabase.table("history").insert([
        {"session_id": session_id, "role": "user", "content": query},
        {"session_id": session_id, "role": "assistant", "content": response_text}
    ]).execute()

    return response_text

@app.get("/test-supabase")
def test_supabase():
    try:
        response = supabase.table("history").select("*").limit(1).execute()
        return {"status": "Connected", "data": response.data}
    except Exception as e:
        return {"status": "Failed", "error": str(e)}

@app.post("/ask")
def ask_raven(request: QueryRequest):
    return {"response": run_rag(request.query, request.session_id)}


@app.post("/process-message/")
async def process_message(request: Request):
    payload = await request.json()
    session_id = payload.get("session_id")
    content = payload.get("content")

    if not session_id or not content:
        raise HTTPException(status_code=400, detail="Missing session_id or content")

    # Call Raven AI chatbot (like in /ask)
    bot_response = run_rag(content, session_id)

    await asyncio.sleep(2)  # Simulate delay (optional)

    try:
        # Use upsert to insert if not exists, otherwise update
        response = supabase.table("messages").upsert([
            {"session_id": session_id, "content": content, "bot_response": bot_response, "status": "completed"}
        ]).execute()

        # ✅ Correctly check for an error
        if response.data is None:  # Check if data is empty
            raise HTTPException(status_code=500, detail="Failed to insert/update message in database")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {"status": "success", "bot_response": bot_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
