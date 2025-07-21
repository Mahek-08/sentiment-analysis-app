import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Load the saved model and vectorizer ---
try:
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    st.error("Model or vectorizer not found. Please run the train.py script first.")
    st.stop()

# --- Preprocessing function (must be the same as in training) ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """Cleans and prepares text for the model."""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# --- Streamlit Web App Interface ---
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🙂")

st.title("Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to find out if it's positive or negative.")

# User input text area
user_input = st.text_area("Movie Review", "I loved this movie, the acting was superb and the story was gripping!")

if st.button("Analyze"):
    if user_input:
        # Preprocess the user input
        cleaned_input = preprocess_text(user_input)
        
        # Vectorize the preprocessed input
        input_vec = vectorizer.transform([cleaned_input])
        
        # Make a prediction
        prediction = model.predict(input_vec)
        prediction_proba = model.predict_proba(input_vec)

        # Display the result
        st.subheader("Analysis Result")
        if prediction[0] == 1:
            st.success(f"Positive Review (Confidence: {prediction_proba[0][1]:.2%})")
        else:
            st.error(f"Negative Review (Confidence: {prediction_proba[0][0]:.2%})")
    else:
        st.warning("Please enter a review to analyze.")

st.markdown("""
---
*This app uses a Logistic Regression model trained on the IMDb movie reviews dataset.*
""")