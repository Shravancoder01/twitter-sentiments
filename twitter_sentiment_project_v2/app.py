import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd

# ----------------- NLTK setup -----------------
nltk.download('punkt')
nltk.download('stopwords')

# ----------------- Load model & vectorizer -----------------
with open("models/vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------- Preprocessing function -----------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def transform_text(text):
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    y = [i for i in tokens if i.isalnum()]
    y = [i for i in y if i not in stop_words]
    y = [ps.stem(i) for i in y]
    return " ".join(y)


# ----------------- Streamlit Setup -----------------
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)

# ---------- Custom CSS for Modern UI ----------
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
            font-family: "Segoe UI", Roboto, sans-serif;
        }
        h1 {
            color: #1a73e8;
            text-align: center;
            font-size: 2.4rem;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        textarea {
            border-radius: 8px !important;
            border: 1px solid #d1d1d1 !important;
            padding: 12px !important;
            font-size: 1rem !important;
        }
        /* Custom Button Style */
        .stButton button {
            background-color: #1a73e8;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #1669c1;
            transform: scale(1.02);
        }
        /* Sentiment Result Container */
        .sentiment-box {
            padding: 1.2rem;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 1.4rem;
            text-align: center;
            font-weight: bold;
        }
        .positive {
            background-color: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #c8e6c9;
        }
        .neutral {
            background-color: #fffde7;
            color: #8a6d3b;
            border: 1px solid #ffe082;
        }
        .negative {
            background-color: #fce4ec;
            color: #c62828;
            border: 1px solid #f5c6cb;
        }
        footer {
            text-align: center;
            font-size: .9rem;
            padding-top: 2rem;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- Header -----------------
st.title("Twitter Sentiment Analyzer")
st.markdown(
    '<p class="subtitle">Analyze tweet sentiment as <b>Positive</b>, <b>Neutral</b>, or <b>Negative</b></p>',
    unsafe_allow_html=True
)

# ----------------- Input -----------------
user_input = st.text_area("Enter your text here:", height=150)

# ----------------- Prediction -----------------
if st.button("Predict Sentiment", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform & vectorize
        transformed_text = transform_text(user_input)
        vector_input = tfidf.transform([transformed_text]).toarray()

        # Predict
        prediction = model.predict(vector_input)[0]
        prediction_proba = None
        try:
            prediction_proba = model.predict_proba(vector_input)[0]  # Fix: take first row for single sample
        except:
            pass

        # Sentiment Mapping
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment_label = sentiment_map.get(prediction, str(prediction))

        # Styled Sentiment Box
        st.markdown(
            f'<div class="sentiment-box {sentiment_label}">'
            f'Sentiment: {sentiment_label.capitalize()}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Show probability bar if available
        if prediction_proba is not None:
            prob_df = pd.DataFrame({
                'Sentiment': ['Negative', 'Neutral', 'Positive'],
                'Probability': prediction_proba  # This is now 1D array, so no error
            })
            st.markdown("#### Prediction Probability")
            st.bar_chart(prob_df.set_index('Sentiment'))

# ----------------- Footer -----------------
st.markdown(
    """
    <footer>
        Developed by <b>Shravan Chafekar</b> | Powered by Python, Scikit-learn & Streamlit
    </footer>
    """,
    unsafe_allow_html=True
)
