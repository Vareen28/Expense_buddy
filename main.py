# main.py
import os
import re
import io
import joblib
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
# ... (Keep CORS middleware) ...

# --- AI Model Loading ---
# REPLACE the Gemini configuration with the OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✅ OpenAI client configured successfully.")
except Exception as e:
    client = None
    print(f"⚠️  Warning: OpenAI client failed to configure. Automatic categorization will be disabled. Error: {e}")

# --- Supabase Client & Pydantic Models ---
# ... (Keep this section the same) ...

# ... (Keep all your classes: SpendingAnalyzer, NudgeEngine, SavingsCalculator) ...

# --- API Endpoints ---
# ... (Keep other endpoints the same) ...



# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Model Loading ---
# Load the pre-trained expense category classifier
try:
    category_classifier = joblib.load('category_classifier.pkl')
    print("✅ Category classifier model loaded successfully.")
except FileNotFoundError:
    print("⚠️  Warning: 'category_classifier.pkl' not found. Automatic categorization will be disabled.")
    category_classifier = None

# --- Supabase Client ---
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

# --- Pydantic Models ---
class Expense(BaseModel):
    amount: float
    merchant: str
    category: Optional[str] = None # Category is now optional; AI will predict if not provided
    date: date
    time: Optional[time] = None
    description: Optional[str] = None
    receipt_url: Optional[str] = None

class ExpenseResponse(BaseModel):
    id: int
    user_id: str
    amount: float
    merchant: str
    category: str
    date: date
    time: Optional[time]
    description: Optional[str]
    receipt_url: Optional[str]
    created_at: datetime

# --- Core Logic Classes ---
class SpendingAnalyzer:
    def __init__(self, supabase_client):
        self.supabase = supabase_client
    
    def get_recent_expenses(self, user_id="demo_user", days=14):
        """Get expenses from the last N days for analysis."""
        from_date = (datetime.now() - timedelta(days=days)).date()
        result = self.supabase.table("expenses").select("*").eq("user_id", user_id).gte("date", from_date.isoformat()).execute()
        return result.data
    
    def analyze_spending_patterns(self, user_id="demo_user"):
        """Analyze spending patterns to identify trends for nudges."""
        expenses = self.get_recent_expenses(user_id)
        category_spending = defaultdict(list)
        late_night_food = []
        
        for expense in expenses:
            category_spending[expense['category']].append(expense)
            if (expense['category'].lower() in ['food', 'dining', 'restaurants'] and 
                expense['time'] and expense['time'] > '22:00:00'):
                late_night_food.append(expense)
        
        total_spent = sum(e['amount'] for e in expenses)
        return {
            "category_totals": {cat: sum(e['amount'] for e in exps) for cat, exps in category_spending.items()},
            "late_night_food": late_night_food,
            "total_expenses": total_spent,
            "avg_daily_spending": total_spent / 14 if expenses else 0
        }

class NudgeEngine:
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def generate_nudges(self, user_id="demo_user"):
        """Generate proactive spending nudges based on analysis."""
        analysis = self.analyzer.analyze_spending_patterns(user_id)
        nudges = []
        
        late_night_spending = sum(e['amount'] for e in analysis['late_night_food'])
        if late_night_spending > 30:
            nudges.append({
                "type": "late_night_food",
                "title": "🌙 Late Night Spending Alert",
                "message": f"You've spent ${late_night_spending:.2f} on late-night food. Consider a weekly meal budget!",
                "action": "set_meal_budget"
            })
        
        if analysis['avg_daily_spending'] > 25:
            weekly_total = analysis['avg_daily_spending'] * 7
            nudges.append({
                "type": "weekly_budget",
                "title": "💰 Weekly Spending Check",
                "message": f"You're averaging ${analysis['avg_daily_spending']:.2f}/day (${weekly_total:.2f}/week). Small cuts could save big!",
                "action": "budget_planner"
            })
        
        return nudges

class SavingsCalculator:
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def calculate_savings_scenarios(self, user_id="demo_user", weekly_saving=20):
        """Calculate what-if savings scenarios."""
        analysis = self.analyzer.analyze_spending_patterns(user_id)
        current_weekly = analysis['avg_daily_spending'] * 7
        
        return {
            "current_spending": {"weekly": current_weekly, "semester": current_weekly * 16},
            "with_savings": {"weekly": current_weekly - weekly_saving, "semester": (current_weekly - weekly_saving) * 16},
            "savings": {"weekly": weekly_saving, "semester": weekly_saving * 16}
        }

# --- Initialize Services ---
analyzer = SpendingAnalyzer(supabase)
nudge_engine = NudgeEngine(analyzer)
savings_calc = SavingsCalculator(analyzer)

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "AI Expense Buddy API is running!"}

@app.get("/health")
async def health_check():
    """Check API and database health."""
    try:
        supabase.table("expenses").select("id", head=True).limit(1).execute()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/expenses/", response_model=dict)
async def create_expense(expense: Expense):
    """Create a new expense, with automatic category prediction if needed."""
    # AI INTEGRATION: Predict category if not provided and the model exists
    if not expense.category and category_classifier:
        try:
            predicted_category = category_classifier.predict([expense.merchant])[0]
            expense.category = predicted_category
        except Exception as e:
            print(f"Model prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Category prediction failed.")
    elif not expense.category:
        raise HTTPException(status_code=400, detail="Category is required as the prediction model is not loaded.")

    try:
        result = supabase.table("expenses").insert({
            "amount": expense.amount, "merchant": expense.merchant, "category": expense.category,
            "date": expense.date.isoformat(), "time": expense.time.isoformat() if expense.time else None,
            "description": expense.description, "receipt_url": expense.receipt_url
        }).execute()
        return {"success": True, "data": result.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/expenses/upload-receipt/")
async def upload_and_parse_receipt(file: UploadFile = File(...)):
    """
    NEW AI FEATURE: Upload a receipt image, perform OCR, and extract expense details.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        receipt_text = pytesseract.image_to_string(image)
        
        def parse_receipt_text(text: str):
            """Uses regex to find merchant, date, and total from raw OCR text."""
            total = 0.0
            # Find the largest monetary value, likely the total
            amounts = re.findall(r'\$?(\d+\.\d{2})', text)
            if amounts:
                total = max([float(a) for a in amounts])
            
            # Simple heuristic for merchant (first non-empty line)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            merchant = lines[0] if lines else "Unknown Merchant"
            
            return {"merchant": merchant, "amount": total, "raw_text": text}
            
        parsed_data = parse_receipt_text(receipt_text)
        return {"filename": file.filename, "extracted_data": parsed_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing receipt: {str(e)}")

@app.get("/expenses/", response_model=List[dict])
async def get_expenses(user_id: str = "demo_user", limit: int = 50):
    """Get a list of recent expenses for a user."""
    try:
        result = supabase.table("expenses").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analysis/{user_id}")
async def get_spending_analysis(user_id: str = "demo_user"):
    """Get a spending analysis for a user."""
    return analyzer.analyze_spending_patterns(user_id)

@app.get("/nudges/{user_id}")
async def get_nudges(user_id: str = "demo_user"):
    """Generate proactive nudges for a user."""
    return {"nudges": nudge_engine.generate_nudges(user_id)}

@app.get("/what-if/{user_id}")
async def what_if_savings(user_id: str = "demo_user", weekly_saving: float = 20):
    """Run a what-if savings scenario."""
    return savings_calc.calculate_savings_scenarios(user_id, weekly_saving)

@app.post("/demo/seed")
async def seed_demo_data():
    """Seed the database with sample data for the demo user."""
    # (Existing seed logic can be kept as is)
    return {"message": "Demo data functionality is present."}

@app.post("/expenses/", response_model=dict)
async def create_expense(expense: Expense):
    """Create a new expense, with automatic category prediction if needed."""
    
    # AI INTEGRATION: Predict category using OpenAI API if not provided
    if not expense.category and client:
        try:
            # Define the categories for the AI
            categories = "'food', 'transport', 'shopping', 'entertainment', 'health', 'bills', 'groceries', 'travel', 'other'"
            
            # Create the API call to OpenAI
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo", # A fast and cost-effective model
                messages=[
                    {"role": "system", "content": f"You are an expert at categorizing expenses. Respond with a single word from these categories only: {categories}."},
                    {"role": "user", "content": f"Categorize this merchant: {expense.merchant}"}
                ]
            )
            
            predicted_category = chat_completion.choices[0].message.content.strip().lower()
            expense.category = predicted_category
            
        except Exception as e:
            print(f"OpenAI API prediction failed: {e}")
            # Fallback to a default category on failure
            expense.category = "other"

    elif not expense.category:
        raise HTTPException(status_code=400, detail="Category is required as the prediction model is not loaded.")

    try:
        # The rest of the function stays the same
        result = supabase.table("expenses").insert({
            "amount": expense.amount, "merchant": expense.merchant, "category": expense.category,
            "date": expense.date.isoformat(), "time": expense.time.isoformat() if expense.time else None,
            "description": expense.description, "receipt_url": expense.receipt_url
        }).execute()
        return {"success": True, "data": result.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))