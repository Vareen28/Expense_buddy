# main.py
import os
import re
import io
from datetime import datetime, timedelta, date, time
from collections import defaultdict
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from openai import OpenAI

# --- Initialization ---
load_dotenv()
app = FastAPI(title="AI Expense Buddy API")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Model Client ---
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✅ OpenAI client configured successfully.")
except Exception as e:
    client = None
    print(f"⚠️ Warning: OpenAI client failed to configure. Automatic categorization will be disabled. Error: {e}")

# --- Supabase Client ---
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

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

    def get_recent_expenses(self, user_id="demo_user", days=14):
        from_date = (datetime.now() - timedelta(days=days)).date()
        result = self.supabase.table("expenses").select("*").eq("user_id", user_id).gte("date", from_date.isoformat()).execute()
        return result.data

    def analyze_spending_patterns(self, user_id="demo_user"):
        expenses = self.get_recent_expenses(user_id)
        category_spending = defaultdict(list)
        total_spent = sum(e['amount'] for e in expenses)
        return {
            "category_totals": {cat: sum(e['amount'] for e in exps) for cat, exps in category_spending.items()},
            "total_expenses": total_spent,
            "avg_daily_spending": total_spent / 14 if expenses else 0
        }

# (Other classes like NudgeEngine and SavingsCalculator can be included here if needed)

# --- Initialize Services ---
analyzer = SpendingAnalyzer(supabase)

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "AI Expense Buddy API is running!"}

@app.get("/health")
async def health_check():
    try:
        supabase.table("expenses").select("id", head=True).limit(1).execute()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/expenses/", response_model=dict)
async def create_expense(expense: Expense):
    if not expense.category and client:
        try:
            categories = "'food', 'transport', 'shopping', 'entertainment', 'health', 'bills', 'groceries', 'travel', 'other'"
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are an expert at categorizing expenses. Respond with a single word from these categories only: {categories}."},
                    {"role": "user", "content": f"Categorize this merchant: {expense.merchant}"}
                ]
            )
            predicted_category = chat_completion.choices[0].message.content.strip().lower()
            expense.category = predicted_category
        except Exception as e:
            print(f"OpenAI API prediction failed: {e}")
            expense.category = "other"
    elif not expense.category:
        raise HTTPException(status_code=400, detail="Category is required as the AI client is not configured.")

    try:
        result = supabase.table("expenses").insert({
            "amount": expense.amount, "merchant": expense.merchant, "category": expense.category,
            "date": expense.date.isoformat(), "time": expense.time.isoformat() if expense.time else None,
            "description": expense.description, "receipt_url": expense.receipt_url
        }).execute()
        return {"success": True, "data": result.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/expenses/", response_model=List[dict])
async def get_expenses(user_id: str = "demo_user", limit: int = 50):
    result = supabase.table("expenses").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
    return result.data

@app.get("/analysis/{user_id}")
async def get_spending_analysis(user_id: str = "demo_user"):
    return analyzer.analyze_spending_patterns(user_id)
