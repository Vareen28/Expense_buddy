import os
import re
import io
import json
import joblib
from datetime import datetime, timedelta, date, time
from collections import defaultdict
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import google.generativeai as genai
import jwt

# --- Initialization ---
load_dotenv()
app = FastAPI(title="AI Expense Buddy API")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Model Loading ---  
# Gemini client configuration
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini client configured successfully.")
except Exception as e:
    model = None
    print(f"⚠️  Warning: Gemini client failed to configure. Error: {e}")

# --- Supabase Client ---
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

# --- Authentication Helper ---
async def get_current_user(authorization: str = Header(None)):
    """Extract and validate user from JWT token."""
    if not authorization or not authorization.startswith("Bearer "):
        # For development, return a demo user
        return "demo_user"
    
    token = authorization.split(" ")[1]
    
    try:
        # Simple decode without verification for development
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload.get("sub", "demo_user")
    except:
        return "demo_user"

# --- Pydantic Models ---
class Expense(BaseModel):
    amount: float
    merchant: str
    category: Optional[str] = None
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

class BillSplitRequest(BaseModel):
    total_amount: float
    participants: List[str]
    description: str

# --- Core Logic Classes ---
class SpendingAnalyzer:
    def __init__(self, supabase_client):
        self.supabase = supabase_client
    
    def get_recent_expenses(self, user_id, days=14):
        """Get expenses from the last N days for analysis."""
        from_date = (datetime.now() - timedelta(days=days)).date()
        try:
            result = self.supabase.table("expenses").select("*").eq("user_id", user_id).gte("date", from_date.isoformat()).execute()
            return result.data
        except:
            return []
    
    def analyze_spending_patterns(self, user_id):
        """Analyze spending patterns to identify trends for nudges."""
        expenses = self.get_recent_expenses(user_id)
        category_spending = defaultdict(list)
        late_night_food = []
        recurring_expenses = defaultdict(int)
        
        for expense in expenses:
            category_spending[expense['category']].append(expense)
            
            # Check for late night food (after 10 PM)
            if (expense['category'].lower() in ['food', 'dining', 'restaurants'] and 
                expense['time'] and expense['time'] > '22:00:00'):
                late_night_food.append(expense)
            
            # Track recurring merchants
            recurring_expenses[expense['merchant']] += 1
        
        total_spent = sum(e['amount'] for e in expenses)
        
        return {
            "category_totals": {cat: sum(e['amount'] for e in exps) for cat, exps in category_spending.items()},
            "late_night_food": late_night_food,
            "recurring_expenses": dict(recurring_expenses),
            "total_expenses": total_spent,
            "avg_daily_spending": total_spent / 14 if expenses else 0,
            "expense_count": len(expenses)
        }

class NudgeEngine:
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def generate_nudges(self, user_id):
        """Generate proactive spending nudges based on analysis."""
        analysis = self.analyzer.analyze_spending_patterns(user_id)
        nudges = []
        
        # Late night food spending nudge
        late_night_spending = sum(e['amount'] for e in analysis['late_night_food'])
        if late_night_spending > 30:
            weekly_meal_budget = late_night_spending * 2  # Suggest double for weekly budget
            nudges.append({
                "type": "late_night_food",
                "title": "🌙 Late Night Spending Alert",
                "message": f"You've spent ${late_night_spending:.2f} on late-night food this week. Consider setting a ${weekly_meal_budget:.0f} weekly meal budget to save money!",
                "action": "set_meal_budget",
                "suggested_budget": weekly_meal_budget,
                "priority": "high"
            })
        
        # High daily spending nudge
        if analysis['avg_daily_spending'] > 25:
            weekly_total = analysis['avg_daily_spending'] * 7
            potential_savings = weekly_total * 0.2  # Suggest 20% reduction
            nudges.append({
                "type": "weekly_budget",
                "title": "💰 Weekly Spending Check",
                "message": f"You're averaging ${analysis['avg_daily_spending']:.2f}/day (${weekly_total:.2f}/week). Reducing by 20% could save ${potential_savings:.2f} weekly!",
                "action": "budget_planner",
                "potential_savings": potential_savings,
                "priority": "medium"
            })
        
        # Recurring subscription nudge
        high_frequency_merchants = {k: v for k, v in analysis['recurring_expenses'].items() if v >= 3}
        if high_frequency_merchants:
            top_merchant = max(high_frequency_merchants, key=high_frequency_merchants.get)
            nudges.append({
                "type": "recurring_expense",
                "title": "🔄 Recurring Expense Detected",
                "message": f"You've spent at {top_merchant} {high_frequency_merchants[top_merchant]} times recently. Consider if this subscription/habit is worth it!",
                "action": "review_subscriptions",
                "merchant": top_merchant,
                "frequency": high_frequency_merchants[top_merchant],
                "priority": "low"
            })
        
        return nudges

class SavingsCalculator:
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def calculate_savings_scenarios(self, user_id, weekly_saving=20):
        """Calculate what-if savings scenarios."""
        analysis = self.analyzer.analyze_spending_patterns(user_id)
        current_weekly = analysis['avg_daily_spending'] * 7
        
        scenarios = {
            "current_spending": {
                "weekly": round(current_weekly, 2),
                "monthly": round(current_weekly * 4.33, 2),
                "semester": round(current_weekly * 16, 2),
                "yearly": round(current_weekly * 52, 2)
            },
            "with_savings": {
                "weekly": round(current_weekly - weekly_saving, 2),
                "monthly": round((current_weekly - weekly_saving) * 4.33, 2),
                "semester": round((current_weekly - weekly_saving) * 16, 2),
                "yearly": round((current_weekly - weekly_saving) * 52, 2)
            },
            "total_savings": {
                "weekly": weekly_saving,
                "monthly": round(weekly_saving * 4.33, 2),
                "semester": round(weekly_saving * 16, 2),
                "yearly": round(weekly_saving * 52, 2)
            }
        }
        
        return scenarios

class CashflowPredictor:
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def predict_future_cashflow(self, user_id, weeks_ahead=4):
        """Predict future spending based on current patterns."""
        analysis = self.analyzer.analyze_spending_patterns(user_id)
        
        # Simple prediction based on average daily spending
        daily_avg = analysis['avg_daily_spending']
        
        predictions = []
        for week in range(1, weeks_ahead + 1):
            week_start = datetime.now() + timedelta(weeks=week-1)
            week_end = week_start + timedelta(days=6)
            
            # Add some variance (±15%) to make it more realistic
            import random
            variance = random.uniform(0.85, 1.15)
            predicted_spending = daily_avg * 7 * variance
            
            predictions.append({
                "week": week,
                "start_date": week_start.strftime("%Y-%m-%d"),
                "end_date": week_end.strftime("%Y-%m-%d"),
                "predicted_spending": round(predicted_spending, 2),
                "confidence": "medium" if week <= 2 else "low"
            })
        
        return {
            "predictions": predictions,
            "based_on_days": 14,
            "avg_daily_spending": round(daily_avg, 2)
        }

# --- Initialize Services ---
analyzer = SpendingAnalyzer(supabase)
nudge_engine = NudgeEngine(analyzer)
savings_calc = SavingsCalculator(analyzer)
cashflow_predictor = CashflowPredictor(analyzer)

# --- Helper Functions ---
async def predict_category_with_gemini(merchant_name: str) -> str:
    """Use Gemini to predict expense category."""
    if not model:
        return "other"
    
    try:
        categories = "food, transport, shopping, entertainment, health, bills, groceries, travel, other"
        
        prompt = f"""
        You are an expert at categorizing expenses for students. 
        Given the merchant name "{merchant_name}", categorize it into one of these categories: {categories}
        
        Respond with only the category name, nothing else.
        
        Examples:
        - McDonald's -> food
        - Uber -> transport
        - Amazon -> shopping
        - Netflix -> entertainment
        - CVS Pharmacy -> health
        - Electric Company -> bills
        - Walmart -> groceries
        - Airbnb -> travel
        
        Merchant: {merchant_name}
        Category:
        """
        
        response = model.generate_content(prompt)
        predicted_category = response.text.strip().lower()
        
        # Validate the response is one of our categories
        valid_categories = ["food", "transport", "shopping", "entertainment", "health", "bills", "groceries", "travel", "other"]
        if predicted_category in valid_categories:
            return predicted_category
        else:
            return "other"
            
    except Exception as e:
        print(f"Gemini API prediction failed: {e}")
        return "other"

def parse_receipt_text_with_gemini(text: str):
    """Use Gemini to parse receipt text."""
    if not model:
        return parse_receipt_text_regex(text)
    
    try:
        prompt = f"""
        Extract expense information from this receipt text:
        
        {text}
        
        Please extract:
        1. Merchant name
        2. Total amount (just the number)
        3. Date (if available, format as YYYY-MM-DD)
        4. Category (food, transport, shopping, entertainment, health, bills, groceries, travel, other)
        
        Respond in JSON format only:
        {{
            "merchant": "merchant name",
            "amount": 0.00,
            "date": "YYYY-MM-DD or null",
            "category": "category"
        }}
        """
        
        response = model.generate_content(prompt)
        parsed_data = json.loads(response.text.strip())
        parsed_data["raw_text"] = text
        return parsed_data
        
    except Exception as e:
        print(f"Gemini parsing failed: {e}")
        return parse_receipt_text_regex(text)

def parse_receipt_text_regex(text: str):
    """Fallback regex-based receipt parsing."""
    total = 0.0
    # Find the largest monetary value, likely the total
    amounts = re.findall(r'\$?(\d+\.\d{2})', text)
    if amounts:
        total = max([float(a) for a in amounts])
    
    # Simple heuristic for merchant (first non-empty line)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    merchant = lines[0] if lines else "Unknown Merchant"
    
    return {
        "merchant": merchant, 
        "amount": total, 
        "date": None,
        "category": "other",
        "raw_text": text
    }

async def generate_bill_split_message(total_amount: float, participants: List[str], description: str):
    """Generate a friendly message for bill splitting."""
    if not model:
        per_person = total_amount / len(participants)
        return f"Hey everyone! We spent ${total_amount:.2f} on {description}. That's ${per_person:.2f} per person. Please send your share! 💰"
    
    try:
        prompt = f"""
        Generate a friendly, casual message to send to roommates/friends for splitting a bill.
        
        Details:
        - Total amount: ${total_amount:.2f}
        - Description: {description}
        - Participants: {', '.join(participants)}
        - Amount per person: ${total_amount / len(participants):.2f}
        
        Make it friendly, clear, and include the amount each person owes. Keep it under 100 words.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        per_person = total_amount / len(participants)
        return f"Hey everyone! We spent ${total_amount:.2f} on {description}. That's ${per_person:.2f} per person. Please send your share! 💰"

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "AI Expense Buddy API is running! 🚀"}

@app.get("/health")
async def health_check():
    """Check API and database health."""
    try:
        supabase.table("expenses").select("id", head=True).limit(1).execute()
        return {"status": "healthy", "database": "connected", "ai": "gemini" if model else "disabled"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/expenses/", response_model=dict)
async def create_expense(expense: Expense, user_id: str = Depends(get_current_user)):
    """Create a new expense, with automatic category prediction if needed."""
    
    # AI INTEGRATION: Predict category using Gemini API if not provided
    if not expense.category:
        expense.category = await predict_category_with_gemini(expense.merchant)

    try:
        result = supabase.table("expenses").insert({
            "user_id": user_id,
            "amount": expense.amount,
            "merchant": expense.merchant,
            "category": expense.category,
            "date": expense.date.isoformat(),
            "time": expense.time.isoformat() if expense.time else None,
            "description": expense.description,
            "receipt_url": expense.receipt_url
        }).execute()
        return {"success": True, "data": result.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/expenses/", response_model=List[dict])
async def get_expenses(user_id: str = Depends(get_current_user), limit: int = 50):
    """Get a list of recent expenses for a user."""
    try:
        result = supabase.table("expenses").select("*").eq("user_id", user_id).order("date", desc=True).limit(limit).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/expenses/upload-receipt/")
async def upload_and_parse_receipt(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    """Upload a receipt image, perform OCR, and extract expense details."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        receipt_text = pytesseract.image_to_string(image)
        
        # Enhanced receipt parsing with Gemini
        parsed_data = parse_receipt_text_with_gemini(receipt_text)
        
        return {"filename": file.filename, "extracted_data": parsed_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing receipt: {str(e)}")

@app.get("/analysis/")
async def get_spending_analysis(user_id: str = Depends(get_current_user)):
    """Get comprehensive spending analysis including nudges and predictions."""
    try:
        # Get spending analysis
        analysis = analyzer.analyze_spending_patterns(user_id)
        
        # Generate proactive nudges
        nudges = nudge_engine.generate_nudges(user_id)
        
        # Get cashflow predictions
        cashflow = cashflow_predictor.predict_future_cashflow(user_id)
        
        return {
            "spending_analysis": analysis,
            "nudges": nudges,
            "cashflow_prediction": cashflow,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/savings-scenarios/")
async def get_savings_scenarios(user_id: str = Depends(get_current_user), weekly_saving: float = 20):
    """Get what-if savings scenarios."""
    try:
        scenarios = savings_calc.calculate_savings_scenarios(user_id, weekly_saving)
        return scenarios
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/bill-split/")
async def generate_bill_split(request: BillSplitRequest, user_id: str = Depends(get_current_user)):
    """Generate a message for splitting bills with roommates."""
    try:
        message = await generate_bill_split_message(
            request.total_amount, 
            request.participants, 
            request.description
        )
        
        per_person = request.total_amount / len(request.participants)
        
        return {
            "message": message,
            "total_amount": request.total_amount,
            "per_person": round(per_person, 2),
            "participants": request.participants,
            "description": request.description
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/nudges/")
async def get_nudges(user_id: str = Depends(get_current_user)):
    """Get proactive spending nudges."""
    try:
        nudges = nudge_engine.generate_nudges(user_id)
        return {"nudges": nudges}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/cashflow-prediction/")
async def get_cashflow_prediction(user_id: str = Depends(get_current_user), weeks: int = 4):
    """Get future cashflow predictions."""
    try:
        prediction = cashflow_predictor.predict_future_cashflow(user_id, weeks)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)