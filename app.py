import streamlit as st
import PyPDF2
import requests
import re
from collections import Counter

# 🔐 Hugging Face API setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}

st.set_page_config(page_title="SmartScribe AI", page_icon="📝")
st.title("📝 SmartScribe AI")
st.subheader("Upload your notes and get instant AI-generated summaries, flashcards & quizzes")

uploaded_file = st.file_uploader("📤 Upload a PDF or TXT file", type=["pdf", "txt"])

# 📄 PDF Text Extractor
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# 📌 Highlight key sentences using basic NLP (regex-based)
def highlight_key_sentences(text, num_sentences=5):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    words = re.findall(r'\w+', text.lower())
    word_freq = Counter(words)

    sentence_scores = {}
    for sentence in sentences:
        score = sum(word_freq[word] for word in re.findall(r'\w+', sentence.lower()))
        sentence_scores[sentence] = score

    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    highlighted_text = text
    for sentence in top_sentences:
        highlighted_text = highlighted_text.replace(sentence, f"**{sentence}**")

    return highlighted_text

# 🤖 Hugging Face Summarization
def summarize_with_huggingface(text):
    # Truncate text to fit within token limits
    truncated_text = text[:700]
    prompt = f"Summarize the following notes as bullet points:\n\n{truncated_text}"
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.5, "max_new_tokens": 300}
    }
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# 🎓 Flashcard Generator
def generate_flashcards(text):
    truncated_text = text[:700]
    prompt = f"Create flashcards from the following notes:\n\n{truncated_text}\n\nFormat:\nQ: ...?\nA: ..."
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.7, "max_new_tokens": 300}
    }
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# ❓ Quiz Generator
def generate_quiz(text):
    truncated_text = text[:700]
    prompt = f"Create a multiple choice quiz from these notes:\n\n{truncated_text}\n\nFormat:\nQ: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer: ..."
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.7, "max_new_tokens": 300}
    }
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# 📄 Extract Text from Uploaded File
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

        # 🟡 Highlight key points
        highlighted = highlight_key_sentences(extracted_text)

        with st.expander("📄 Show Extracted Text with Highlights"):
            st.markdown(highlighted)

        # ✨ Summarize Button
        if st.button("✨ Summarize Notes"):
            with st.spinner("Summarizing..."):
                try:
                    summary = summarize_with_huggingface(extracted_text)
                    st.subheader("🧠 Bullet Point Summary")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

        # 📚 Flashcards Button
        if st.button("🗂️ Generate Flashcards"):
            with st.spinner("Creating flashcards..."):
                try:
                    flashcards = generate_flashcards(extracted_text)
                    st.subheader("🧾 Flashcards")
                    st.markdown(flashcards)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

        # ❓ Quiz Button
        if st.button("📝 Generate Quiz"):
            with st.spinner("Generating quiz..."):
                try:
                    quiz = generate_quiz(extracted_text)
                    st.subheader("📋 Quiz")
                    st.markdown(quiz)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

