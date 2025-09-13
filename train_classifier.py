import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# 1. Load Data
df = pd.read_csv('transactions.csv')

# 2. Define Features (X) and Target (y)
X = df['merchant']
y = df['category']

# 3. Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create a model pipeline
#    - TfidfVectorizer: Converts text to numerical vectors
#    - LogisticRegression: The classification model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# 5. Train the model
pipeline.fit(X_train, y_train)

# 6. Evaluate the model (optional but recommended)
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# 7. Save the trained model for use in your app
joblib.dump(pipeline, 'category_classifier.pkl')
print("Model saved to category_classifier.pkl")