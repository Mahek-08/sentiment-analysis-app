# Sentiment Analysis Web App

A machine learning app that classifies movie reviews as "Positive" or "Negative". This project demonstrates an end-to-end NLP workflow, from data cleaning and model training to deployment as an interactive Streamlit web app.


## Key Features

* **Data Processing:** Cleans and preprocesses raw text from the IMDb dataset using Pandas and NLTK, including stopword removal and stemming.
* **ML Model:** Uses a Scikit-learn `Logistic Regression` model with `TF-IDF` vectorization to achieve ~88% accuracy on the test set.
* **Interactive Web App:** A clean UI built with Streamlit provides real-time predictions with confidence scores.

## Technologies

* **Modeling & Data Manipulation:** Python, Scikit-learn, Pandas, NLTK
* **Web Frontend:** Streamlit
* **Dataset Source:** IMDb Movie Reviews

## Local Setup and Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/sentiment-analysis-app.git](https://github.com/YOUR_USERNAME/sentiment-analysis-app.git)
cd sentiment-analysis-app
```

### 2. Install Dependencies

Install the required Python libraries using the `requirements.txt` file. (See final section for creation instructions).

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

The model requires the **IMDb Dataset of 50K Movie Reviews**.

* **Download Link:** [Kaggle IMDb Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* Place the downloaded `IMDB Dataset.csv` file in the project's root directory.

### 4. Train the Model

Run the training script to preprocess the data and save the model artifacts.

```bash
python train.py
```

## Usage

Launch the Streamlit web application with the following command. The app will open in your browser at `http://localhost:8501`.

```bash
streamlit run app.py
```

## Project Structure

```
.
├── app.py                  # The Streamlit web application script
├── train.py                # Script for training the model and vectorizer
├── sentiment_model.joblib  # Saved trained Logistic Regression model
├── tfidf_vectorizer.joblib # Saved TF-IDF vectorizer object
├── requirements.txt        # List of Python dependencies for reproducibility
├── .gitignore              # Specifies files to be ignored by Git
└── README.md               # This file
```

### Creating `requirements.txt`

For project reproducibility, it's crucial to have a `requirements.txt` file. Generate it automatically from your environment by running this command in your terminal:

```bash
pip freeze > requirements.txt
