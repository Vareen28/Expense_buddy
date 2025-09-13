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
    print(f"⚠️ Warning: OpenAI client failed to configure. AI features will be disabled. Error: {e}")

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
    """NEW: AI-powered engine to generate proactive financial advice."""
    def __init__(self, analyzer, ai_client):
        self.analyzer = analyzer
        self.client = ai_client

    def generate_nudges(self, user_id="demo_user"):
        if not self.client:
            return {"error": "AI client is not configured. Nudges are unavailable."}

        analysis = self.analyzer.analyze_spending_patterns(user_id)
        
        # Don't generate nudges if there's no recent activity
        if analysis['transaction_count'] < 3:
            return []

        try:
            prompt = f"""
            As a friendly financial advisor for a student, analyze the following spending data from the last 14 days and generate one encouraging, actionable nudge.
            The user is on a budget. If spending is high, suggest a specific, small change. If it's low, be encouraging.
            Keep the message concise (under 280 characters).
            
            Data:
            - Total Spent: ${analysis['total_expenses']:.2f}
            - Average Daily Spending: ${analysis['avg_daily_spending']:.2f}
            - Spending by Category: {analysis['category_totals']}

            Generate a JSON object with "title" and "message".
            """
            
            chat_completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a helpful and concise financial assistant for students."},
                    {"role": "user", "content": prompt}
                ]
            )
            nudge = chat_completion.choices[0].message.content
            return [nudge] # Return as a list for consistency
        except Exception as e:
            print(f"Nudge generation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate AI nudge.")


class SavingsCalculator:
    """NEW: Engine to calculate 'what-if' savings scenarios."""
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def calculate_savings_scenarios(self, user_id="demo_user", weekly_saving_goal=25):
        analysis = self.analyzer.analyze_spending_patterns(user_id)
        current_weekly_spend = analysis['avg_daily_spending'] * 7
        
        projected_weekly_spend = current_weekly_spend - weekly_saving_goal

        return {
            "scenario_details": {
                "weekly_saving_goal": weekly_saving_goal,
                "based_on_transactions_in_last_days": 14
            },
            "current_trajectory": {
                "weekly_spend": round(current_weekly_spend, 2),
                "monthly_spend": round(current_weekly_spend * 4.33, 2),
                "semester_spend (16 weeks)": round(current_weekly_spend * 16, 2)
            },
            "new_trajectory_with_savings": {
                "weekly_spend": round(projected_weekly_spend, 2),
                "monthly_spend": round(projected_weekly_spend * 4.33, 2),
                "semester_spend (16 weeks)": round(projected_weekly_spend * 16, 2)
            },
            "total_savings": {
                "per_week": weekly_saving_goal,
                "per_month": round(weekly_saving_goal * 4.33, 2),
                "per_semester": weekly_saving_goal * 16
            }
        }


# --- Initialize Services ---
analyzer = SpendingAnalyzer(supabase)
nudge_engine = NudgeEngine(analyzer, client)
savings_calc = SavingsCalculator(analyzer)


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "AI Expense Buddy API is running! Append /docs to the URL to see the API documentation."}

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

@app.get("/nudges/{user_id}", response_model=dict)
async def get_nudges(user_id: str = "demo_user"):
    """NEW: Get an AI-powered financial nudge based on recent spending."""
    nudges = nudge_engine.generate_nudges(user_id)
    return {"nudges": nudges}

@app.get("/what-if/{user_id}", response_model=dict)
async def get_what_if_scenario(user_id: str = "demo_user", weekly_saving: int = 25):
    """NEW: Run a 'what-if' scenario to see the impact of saving money."""
    scenario = savings_calc.calculate_savings_scenarios(user_id, weekly_saving)
    return scenario
