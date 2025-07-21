import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Download NLTK data (only need to do this once) ---
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    print("Download complete.")

# --- 1. Load and Prepare the Dataset ---
print("Loading dataset...")
# This dataset is large. The original file can be found on Kaggle:
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# For this script, ensure 'IMDB Dataset.csv' is in the same directory.
# I will proceed assuming the file is present. Due to my limitations, I cannot download it myself.
try:
    df = pd.read_csv('IMDB Dataset.csv')
except FileNotFoundError:
    print("\nERROR: 'IMDB Dataset.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the same directory as this script.")
    print("Link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    exit()

print("Dataset loaded successfully.")

# --- 2. Preprocess the Text Data ---
print("Preprocessing text data...")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """Cleans and prepares the text for modeling."""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # Tokenize and remove stopwords
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing to the review column
df['cleaned_review'] = df['review'].apply(preprocess_text)
print("Preprocessing complete.")

# --- 3. Split Data and Vectorize ---
# Map labels to 0 and 1
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Define features (X) and target (y)
X = df['cleaned_review']
y = df['sentiment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF Vectorizer
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Vectorizing complete.")

# --- 4. Train the Model ---
print("Training the model...")
model = LogisticRegression()
model.fit(X_train_vec, y_train)
print("Model training complete.")

# --- 5. Evaluate the Model ---
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# --- 6. Save the Model and Vectorizer ---
print("Saving model and vectorizer...")
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("\nTraining complete. The model and vectorizer have been saved as:")
print("- sentiment_model.joblib")
print("- tfidf_vectorizer.joblib")