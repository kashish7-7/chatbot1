import os
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# Load env
load_dotenv()

# Import your app from app.py
from app import app as fastapi_app

# Convert ASGI → WSGI for PythonAnywhere
from mangum import Mangum
handler = Mangum(fastapi_app)
