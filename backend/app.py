import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / '.env'

print(f"Looking for .env at: {ENV_PATH}")
print(f".env exists: {ENV_PATH.exists()}")

load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print(f"API Key loaded: {GROQ_API_KEY[:10] if GROQ_API_KEY else 'None'}...")

if not GROQ_API_KEY:
    raise ValueError("API key for Groq is missing. Please set GROQ_API_KEY in .env file")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(data: Query):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": data.question}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Send POST request to /ask or /chat"}

class UserInput(BaseModel):
    message: str
    role: str = "user"
    conversation_id: str

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a useful AI assistant."}
        ]
        self.active: bool = True

conversations: Dict[str, Conversation] = {}

def query_groq_api(conversation: Conversation) -> str:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation.messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
        )
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq Error: {str(e)}")

def get_or_create_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation()
    return conversations[conversation_id]

@app.post("/chat/")
async def chat(input: UserInput):
    conversation = get_or_create_conversation(input.conversation_id)
    if not conversation.active:
        raise HTTPException(
            status_code=400,
            detail="Chat session ended. Start a new session."
        )
    try:
        conversation.messages.append({
            "role": input.role,
            "content": input.message
        })
        response = query_groq_api(conversation)
        conversation.messages.append({
            "role": "assistant",
            "content": response
        })
        return {
            "response": response,
            "conversation_id": input.conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)