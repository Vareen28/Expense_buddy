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
import google.generativeai as genai
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

# --- AI Model Client (Gemini) ---
try:
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
    """Fetches the JWKS from Clerk for token validation."""
    response = requests.get(CLERK_JWKS_URL)
    response.raise_for_status()
    return response.json()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Validates a Clerk JWT and returns the user ID."""
    jwks = get_jwks()
    try:
        unverified_header = jwt.get_unverified_header(token)
        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {"kty": key["kty"], "kid": key["kid"], "use": key["use"], "n": key["n"], "e": key["e"]}
        
        if rsa_key:
            payload = jwt.decode(token, rsa_key, algorithms=["RS256"])
            user_id = payload.get("sub")
            if user_id is None:
                raise HTTPException(status_code=401, detail="User ID not found in token")
            return {"id": user_id}
        
        raise HTTPException(status_code=401, detail="Unable to find appropriate key")
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Could not validate credentials: {e}")

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
    def __init__(self, supabase_client):
        self.supabase = supabase_client

    def get_recent_expenses(self, user_id: str, days=14):
        from_date = (datetime.now() - timedelta(days=days)).date()
        result = self.supabase.table("expenses").select("*").eq("user_id", user_id).gte("date", from_date.isoformat()).execute()
        return result.data

    def analyze_spending_patterns(self, user_id: str):
        expenses = self.get_recent_expenses(user_id)
        category_spending = defaultdict(float)
        for expense in expenses:
            category_spending[expense['category']] += expense['amount']
        
        total_spent = sum(e['amount'] for e in expenses)
        return {
            "category_totals": dict(category_spending),
            "total_expenses": total_spent,
            "avg_daily_spending": total_spent / 14 if expenses else 0,
            "transaction_count": len(expenses)
        }

class NudgeEngine:
    def __init__(self, analyzer, ai_model):
        self.analyzer = analyzer
        self.model = ai_model

    def generate_nudges(self, user_id: str):
        if not self.model: return {"error": "AI client is not configured."}
        analysis = self.analyzer.analyze_spending_patterns(user_id)
        if analysis['transaction_count'] < 3: return []
        
        try:
            prompt = f"As a friendly financial advisor, analyze the user's spending ({analysis}) and generate one concise, actionable nudge in JSON format with a 'title' and 'message'."
            response = self.model.generate_content(prompt)
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
            return [json.loads(cleaned_text)]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate AI nudge: {e}")

class SavingsCalculator:
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def calculate_savings_scenarios(self, user_id: str, weekly_saving_goal=25):
        analysis = self.analyzer.analyze_spending_patterns(user_id)
        current_weekly_spend = analysis['avg_daily_spending'] * 7
        projected_weekly_spend = current_weekly_spend - weekly_saving_goal
        return {
            "current_trajectory": {"weekly_spend": round(current_weekly_spend, 2), "semester_spend": round(current_weekly_spend * 16, 2)},
            "new_trajectory_with_savings": {"weekly_spend": round(projected_weekly_spend, 2), "semester_spend": round(projected_weekly_spend * 16, 2)},
            "total_savings": {"per_week": weekly_saving_goal, "per_semester": weekly_saving_goal * 16}
        }

# --- Initialize Services ---
analyzer = SpendingAnalyzer(supabase)
nudge_engine = NudgeEngine(analyzer, model)
savings_calc = SavingsCalculator(analyzer)

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "AI Expense Buddy API is running!"}

@app.post("/expenses/")
async def create_expense(expense: Expense, user: dict = Depends(get_current_user)):
    user_id = user["id"]
    if not expense.category and model:
        try:
            categories = "'food', 'transport', 'shopping', 'entertainment', 'health', 'bills', 'groceries', 'travel', 'other'"
            prompt = f"Categorize this merchant: '{expense.merchant}'. Respond with one word from: {categories}."
            response = model.generate_content(prompt)
            expense.category = response.text.strip().lower()
        except Exception as e:
            print(f"Gemini API prediction failed: {e}")
            expense.category = "other"
    
    try:
        expense_data = expense.model_dump()
        expense_data["user_id"] = user_id
        result = supabase.table("expenses").insert(expense_data).execute()
        return {"success": True, "data": result.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/expenses/", response_model=List[dict])
async def get_expenses(user: dict = Depends(get_current_user), limit: int = 50):
    result = supabase.table("expenses").select("*").eq("user_id", user["id"]).order("created_at", desc=True).limit(limit).execute()
    return result.data

@app.get("/analysis/")
async def get_spending_analysis(user: dict = Depends(get_current_user)):
    return analyzer.analyze_spending_patterns(user["id"])

@app.get("/nudges/", response_model=dict)
async def get_nudges(user: dict = Depends(get_current_user)):
    return {"nudges": nudge_engine.generate_nudges(user["id"])}

@app.get("/what-if/", response_model=dict)
async def get_what_if_scenario(user: dict = Depends(get_current_user), weekly_saving: int = 25):
    return savings_calc.calculate_savings_scenarios(user["id"], weekly_saving)
