Of course. Here is a complete `README.md` file for your project. You can copy and paste this content into a file named `README.md` in your project's root directory.

-----

# AI Expense Buddy API 🤖💰

This is the backend API for the AI Expense Buddy, a smart personal finance application designed to help users manage their money effectively. It leverages the OpenAI API to automatically categorize expenses, parse receipts using OCR, and provide intelligent financial insights.

-----

## Features ✨

  * **AI-Powered Expense Categorization:** Automatically assigns a category to a new expense by analyzing the merchant name using the OpenAI (GPT) API.
  * **Receipt Parsing:** A dedicated endpoint to upload a receipt image. It uses Optical Character Recognition (OCR) to extract the merchant and total amount.
  * **Expense Tracking:** Full CRUD (Create, Read, Update, Delete) functionality for managing expenses.
  * **Spending Analysis:** An endpoint to analyze recent spending patterns by category.
  * **Cloud Database:** Uses Supabase (PostgreSQL) for reliable and scalable data storage.

-----

## Tech Stack 🛠️

  * **Backend Framework:** FastAPI
  * **Database:** Supabase
  * **AI for Categorization:** OpenAI API
  * **AI for Receipt Parsing:** Tesseract OCR (via `pytesseract`)
  * **Deployment:** Ready for services like Render, Vercel, or Heroku.

-----

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

  * Python 3.9+
  * A Supabase account
  * An OpenAI API key

### 1\. Clone the Repository

```bash
git clone https://your-repository-url.git
cd your-project-directory
```

### 2\. Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
python -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Activate it (Windows)
.\venv\Scripts\activate
```

### 3\. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4\. Configure Environment Variables

Create a file named `.env` in the root of your project and add the following variables. These are your secret keys and should not be committed to Git.

```
SUPABASE_URL="your_supabase_project_url"
SUPABASE_ANON_KEY="your_supabase_anon_key"
OPENAI_API_KEY="your_sk-openai_api_key"
```

### 5\. Run the Local Server

Start the FastAPI application using Uvicorn. The `--reload` flag will automatically restart the server when you make changes to the code.

```bash
uvicorn main:app --reload
```

The API will now be running at `http://127.0.0.1:8000`.

-----

## API Endpoints 📖

Here are the main endpoints available in the API. You can also view interactive documentation by navigating to `http://127.0.0.1:8000/docs` while the server is running.

| Endpoint                     | Method | Description                                                                   |
| :--------------------------- | :----- | :---------------------------------------------------------------------------- |
| `/`                          | `GET`    | Root endpoint to confirm the API is running.                                  |
| `/health`                    | `GET`    | Checks the health of the API and its connection to the database.              |
| `/expenses/`                 | `POST`   | Creates a new expense. If `category` is omitted, the AI will predict it.      |
| `/expenses/`                 | `GET`    | Retrieves a list of recent expenses for the user.                             |
| `/expenses/upload-receipt/`  | `POST`   | Upload a receipt image (`.png`, `.jpg`, etc.) to extract its details via OCR. |
| `/analysis/{user_id}`        | `GET`    | Provides a summary of spending patterns for the specified user.               |

-----

## Deployment ☁️

This application is ready to be deployed on any platform that supports Python web services (like Render).

1.  Push your code to a GitHub repository.
2.  Connect your repository to the deployment service.
3.  Set the **Environment Variables** (`SUPABASE_URL`, `SUPABASE_ANON_KEY`, `OPENAI_API_KEY`) in your service's dashboard.
4.  Use the following commands for the build process:
      * **Build Command:** `pip install -r requirements.txt`
      * **Start Command:** `uvicorn main:app --host=0.0.0.0 --port=10000`
