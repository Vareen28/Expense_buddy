# main.py
import os
import re
import io
import json
from datetime import datetime, timedelta, date, time
from collections import defaultdict
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import google.generativeai as genai # Replaced OpenAI with Google Generative AI
from jose import jwt
from jose.exceptions import JWTError
import requests

# --- Initialization ---
load_dotenv()
app = FastAPI(title="AI Expense Buddy API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Model Client (Switched to Gemini) ---
try:
    # Use GOOGLE_API_KEY from your environment variables
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Google Gemini client configured successfully.")
except Exception as e:
    model = None
    print(f"⚠️ Warning: Google Gemini client failed to configure. AI features will be disabled. Error: {e}")

# --- Supabase and Authentication ---
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")

def get_jwks():
    response = requests.get(CLERK_JWKS_URL)
    response.raise_for_status()
    return response.json()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # This authentication logic remains the same
    pass

# --- Pydantic Models ---
class Expense(BaseModel):
    amount: float
    merchant: str
    category: Optional[str] = None
    date: date
    time: Optional[time] = None
    description: Optional[str] = None
    receipt_url: Optional[str] = None

# --- Core Logic Classes ---
class SpendingAnalyzer:
    # This class remains the same
    pass

class NudgeEngine:
    def __init__(self, analyzer, ai_model):
        self.analyzer = analyzer
        self.model = ai_model

    def generate_nudges(self, user_id: str):
        if not self.model:
            return {"error": "AI client is not configured."}
        
        analysis = self.analyzer.analyze_spending_patterns(user_id)
        if analysis['transaction_count'] < 3:
            return []
        
        try:
            prompt = f"""
            As a friendly financial advisor for a student, analyze the following spending data and generate one encouraging, actionable nudge.
            Keep the message concise.
            Data:
            - Total Spent: ${analysis['total_expenses']:.2f}
            - Avg Daily Spend: ${analysis['avg_daily_spending']:.2f}
            - Categories: {analysis['category_totals']}
            
            Respond with a valid JSON object in this exact format: {{"title": "string", "message": "string"}}
            """
            
            # Updated to use Gemini's generate_content method
            response = self.model.generate_content(prompt)
            # Clean up the response to ensure it's valid JSON
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
            nudge = json.loads(cleaned_text)
            return [nudge]

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate AI nudge: {e}")

# ... (SavingsCalculator and initialization of services)

# --- API Endpoints ---
@app.post("/expenses/")
async def create_expense(expense: Expense, user: dict = Depends(get_current_user)):
    if not expense.category and model:
        try:
            categories = "'food', 'transport', 'shopping', 'entertainment', 'health', 'bills', 'groceries', 'travel', 'other'"
            prompt = f"Categorize the following merchant into one of these exact categories: {categories}. Respond with only the single category word. Merchant: '{expense.merchant}'"
            
            # Updated to use Gemini's generate_content method
            response = model.generate_content(prompt)
            expense.category = response.text.strip().lower()

        except Exception as e:
            print(f"Gemini API prediction failed: {e}")
            expense.category = "other"
    elif not expense.category:
        raise HTTPException(status_code=400, detail="Category is required.")
    
    # The rest of the function remains the same
    pass

# ... (The rest of your endpoints remain the same)
