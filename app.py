import streamlit as st
import requests
import re
import nltk
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Ensure required NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Streamlit UI Setup
st.set_page_config(page_title="SmartScribe AI", page_icon="📝", layout="wide")

# Custom CSS for sleek look
st.markdown("""
    <style>
        body {
            background-color: #f7f7f7;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 30px;
            border: none;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stTextInput input {
            border-radius: 8px;
            padding: 12px 20px;
            font-size: 16px;
            border: 1px solid #ddd;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }
        .stTextInput input:focus {
            outline: none;
            border-color: #4CAF50;
        }
        .stMarkdown {
            font-size: 16px;
            line-height: 1.6;
        }
        .stFileUploader {
            border: 2px dashed #4CAF50;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }
        .stFileUploader:hover {
            background-color: #f0f0f0;
        }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📝 SmartScribe AI")
st.subheader("Upload your notes and get instant AI-generated flashcards and text insights!")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a section", 
                           ["📄 Extracted Text", "✨ Generate Flashcards", 
                            "📊 Text Analytics", "📊 Sentiment Analysis"])

# File Upload Section
uploaded_file = st.file_uploader("📤 Upload a PDF or TXT file", type=["pdf", "txt"])

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    # Your code for extracting text from PDF
    pass

# Flashcard generation function (simplified for example)
def generate_flashcards(text):
    # You can replace this with actual flashcard logic
    flashcards = text.split('\n')
    return flashcards

# Sentiment Analysis function (simplified for example)
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"
    return sentiment, blob.sentiment.polarity

# Handling file upload and processing
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "txt":
        extracted_text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type.")
    
    if extracted_text.strip() == "":
        st.warning("⚠️ Couldn't extract any text. Try another file.")
    else:
        st.success("✅ Text extracted successfully!")

        # Display Extracted Text
        if app_mode == "📄 Extracted Text":
            st.subheader("📄 Extracted Text")
            st.markdown(f"<div style='background-color: white; padding: 20px; border-radius: 10px;'>{extracted_text}</div>", unsafe_allow_html=True)

        # Flashcards Section
        if app_mode == "✨ Generate Flashcards":
            st.subheader("🧠 Flashcards")
            with st.spinner("Generating flashcards..."):
                try:
                    flashcards = generate_flashcards(extracted_text)  # This is where you'd call your flashcard function
                    for idx, flashcard in enumerate(flashcards):
                        st.markdown(f"""
                            <div style="background-color:#e8f5e9; padding: 20px; 
                            border-radius: 10px; margin-bottom: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                                <strong>🧠 Flashcard {idx+1}:</strong><br>{flashcard}
                            </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

        # Sentiment Analysis Section
        if app_mode == "📊 Sentiment Analysis":
            st.subheader("📊 Sentiment Analysis")
            sentiment, polarity = analyze_sentiment(extracted_text)
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Polarity Score:** {polarity:.2f}")
            sentiment_color = "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "gray"
            st.markdown(f'<p style="color:{sentiment_color}; font-size: 18px;">{sentiment}</p>', unsafe_allow_html=True)

