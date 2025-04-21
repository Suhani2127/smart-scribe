import streamlit as st
import PyPDF2
import requests
import re
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# 🔐 Hugging Face API setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}

nltk.download('punkt')

st.set_page_config(page_title="SmartScribe AI", page_icon="📝")
st.title("📝 SmartScribe AI")
st.subheader("Upload notes → Get summary, quizzes, flashcards & highlights ✨")

uploaded_file = st.file_uploader("📤 Upload a PDF or TXT file", type=["pdf", "txt"])


# 🔍 PDF Text Extractor
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# ✨ Hugging Face Summarizer
def summarize_with_huggingface(text):
    prompt = f"Summarize the following notes into bullet points:\n\n{text[:700]}"
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.5, "max_new_tokens": 300}
    }

    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()[0]["generated_text"]
        return "\n".join(re.findall(r"[-•*].+", result))  # Extract only bullet lines
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


# 🧠 Flashcard Generator
def generate_flashcards(text):
    prompt = f"Create flashcards (Q&A pairs) from the following:\n\n{text[:700]}"
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.5, "max_new_tokens": 300}
    }

    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


# 📌 Highlight key sentences using basic NLP (based on term frequency)
def highlight_key_sentences(text, num_highlights=5):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_highlights:
        return text  # Not enough to highlight

    cv = CountVectorizer(stop_words='english')
    X = cv.fit_transform(sentences)
    scores = np.array(X.sum(axis=1)).flatten()

    top_indices = scores.argsort()[-num_highlights:][::-1]
    highlights = set(sentences[i] for i in top_indices)

    highlighted_text = ""
    for sentence in sentences:
        if sentence in highlights:
            highlighted_text += f"**🟡 {sentence.strip()}**  \n"
        else:
            highlighted_text += sentence.strip() + "  \n"
    return highlighted_text


# 🚀 Main App Logic
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "txt":
        extracted_text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type.")
        st.stop()

    if extracted_text.strip() == "":
        st.warning("⚠️ Couldn't extract any text. Try another file.")
    else:
        st.success("✅ Text extracted successfully!")

        # Show Extracted Text with Highlights
        st.subheader("📄 Highlighted Key Points in Extracted Text")
        highlighted = highlight_key_sentences(extracted_text)
        st.markdown(highlighted)

        # ✨ Summarize Button
        if st.button("🧠 Summarize Notes"):
            with st.spinner("Summarizing..."):
                try:
                    summary = summarize_with_huggingface(extracted_text)
                    st.subheader("📌 Bullet Point Summary")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Something went wrong while summarizing: {e}")

        # 📚 Flashcards & Quiz
        if st.button("🗂️ Generate Flashcards / Quiz"):
            with st.spinner("Generating flashcards..."):
                try:
                    cards = generate_flashcards(extracted_text)
                    st.subheader("🎴 Flashcards / Quiz Questions")
                    st.markdown(cards)
                except Exception as e:
                    st.error(f"Something went wrong while generating flashcards: {e}")

