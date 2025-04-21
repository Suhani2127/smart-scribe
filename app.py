import streamlit as st
import PyPDF2
import requests
import re
from collections import Counter
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Ensure required NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.tokenize import word_tokenize, sent_tokenize

# Hugging Face API setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}

# Text Analytics Helpers
def clean_and_tokenize(text):
    stop_words = {
        "the", "and", "is", "in", "it", "of", "to", "for", "a", "an", "on", "with", "as", "by", "this",
        "that", "from", "or", "at", "are", "be", "was", "were", "has", "have", "had", "not"
    }
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())  # only words with 3+ letters
    return [token for token in tokens if token not in stop_words]


def get_top_keywords(text, n=10):
    tokens = clean_and_tokenize(text)
    counter = Counter(tokens)
    return counter.most_common(n)


def split_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text.strip())


def get_top_sentences(text, n=5):
    sentences = split_into_sentences(text)
    if len(sentences) == 0:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    top_indices = sentence_scores.argsort()[-n:][::-1]
    return [sentences[i] for i in top_indices]


# Split the text into smaller chunks to avoid token limit issues
def split_text(text, max_chunk_size=1024):
    sentences = re.split(r'(?<=\.)\s', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# Generate Flashcards using Hugging Face API
def generate_flashcards_with_huggingface(text_chunk):
    prompt = f"Generate flashcards based on the following notes:\n\n{text_chunk}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.5,
            "max_new_tokens": 150
        }
    }

    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# PDF Text Extractor
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    return sentiment, polarity

# Streamlit UI with Custom CSS
st.set_page_config(page_title="SmartScribe AI", page_icon="📝")
st.title("📝 SmartScribe AI")
st.subheader("Upload your notes and get instant AI-generated flashcards and text insights!")

# Custom CSS for UI
st.markdown("""
    <style>
        body {
            background-color: #f7f7f7;
            font-family: 'Helvetica Neue', sans-serif;
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
        .stSidebar .stRadio button {
            border: none;
            background: none;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            color: #333;
            display: block;
            margin-bottom: 10px;
            text-align: left;
            width: 100%;
        }
        .stSidebar .stRadio button:hover {
            background-color: #f4f4f4;
        }
        .stSidebar .stRadio div {
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a section", 
                           ["📄 Extracted Text", "✨ Generate Flashcards", 
                            "📊 Text Analytics", "📊 Sentiment Analysis"])

# File Upload Section
uploaded_file = st.file_uploader("📤 Upload a PDF or TXT file", type=["pdf", "txt"])
extracted_text = ""

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

        # Show Extracted Text
        if app_mode == "📄 Extracted Text":
            st.subheader("📄 Extracted Text")
            st.write(extracted_text)

        # Flashcards Section
       
# Flashcards Section
if app_mode == "✨ Generate Flashcards":
    with st.spinner("Generating flashcards..."):
        try:
            chunks = split_text(extracted_text)
            flashcards = []
            for chunk in chunks:
                flashcards_chunk = generate_flashcards_with_huggingface(chunk)
                flashcards.append(flashcards_chunk)

            st.subheader("🧠 Flashcards")
            # Display flashcards in a grid layout
            cols = st.columns(2)  # Create two columns for a compact layout
            for idx, flashcard in enumerate(flashcards):
                flashcard_list = flashcard.split("\n")
                with cols[idx % 2]:
                    for card in flashcard_list:
                        if card.strip():
                            st.markdown(
                                f"""
                                <div style="background-color:#e8f5e9; padding: 10px; 
                                border-radius: 10px; margin-bottom: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                                    <strong>🧠 Flashcard:</strong><br><span style="color:black;">{card.strip()}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")


        # Text Analytics Section
        if app_mode == "📊 Text Analytics":
            st.subheader("🔑 Top Keywords")
            top_keywords = get_top_keywords(extracted_text, n=10)
            for word, freq in top_keywords:
                st.markdown(f"- **{word}** ({freq} times)")

            st.subheader("📌 Top Sentences")
            top_sentences = get_top_sentences(extracted_text, n=5)
            for sent in top_sentences:
                st.markdown(f"> {sent}")

        # Sentiment Analysis Section
        if app_mode == "📊 Sentiment Analysis":
            st.subheader("📊 Sentiment Analysis")
            sentiment, polarity = analyze_sentiment(extracted_text)

            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Polarity Score:** {polarity:.2f}")

            sentiment_color = "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "gray"
            st.markdown(f'<p style="color:{sentiment_color}; font-size: 18px;">{sentiment}</p>', unsafe_allow_html=True)
