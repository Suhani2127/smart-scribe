import streamlit as st
import PyPDF2
import requests
import re

# 🔐 Hugging Face API setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}

st.set_page_config(page_title="SmartScribe AI", page_icon="📝")
st.title("📝 SmartScribe AI")
st.subheader("Upload your notes and get instant AI-generated summaries & quizzes (free & open-source powered)")

uploaded_file = st.file_uploader("📤 Upload a PDF or TXT file", type=["pdf", "txt"])

# 🔍 PDF Text Extractor
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Split the text into smaller chunks to avoid token limit issues
def split_text(text, max_chunk_size=1024):
    # Split text into chunks of size max_chunk_size (taking into account token limits)
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

# 🤖 Summarization using Hugging Face API
def summarize_with_huggingface(text_chunk):
    prompt = f"Summarize the following notes:\n\n{text_chunk}"
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

# 🤖 Generate Quizzes using Hugging Face API
def generate_quiz_with_huggingface(text_chunk):
    prompt = f"Generate a multiple-choice quiz based on the following notes:\n\n{text_chunk}"
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
        with st.expander("📄 Show Extracted Text"):
            st.write(extracted_text)

        # ✨ Summarize Button
        if st.button("✨ Summarize Notes"):
            with st.spinner("Summarizing..."):
                try:
                    # Split text into manageable chunks for summary
                    chunks = split_text(extracted_text)

                    summaries = []
                    for chunk in chunks:
                        summary_chunk = summarize_with_huggingface(chunk)
                        summaries.append(summary_chunk)

                    st.subheader("🧠 Summary")
                    # Display summary as bullet points
                    for summary in summaries:
                        st.markdown(f"- {summary.strip()}")

                except Exception as e:
                    st.error(f"Something went wrong: {e}")

        # ✨ Quiz Button
        if st.button("✨ Generate Quiz"):
            with st.spinner("Generating quiz..."):
                try:
                    # Split text into manageable chunks for quiz
                    chunks = split_text(extracted_text)

                    quizzes = []
                    for chunk in chunks:
                        quiz_chunk = generate_quiz_with_huggingface(chunk)
                        quizzes.append(quiz_chunk)

                    st.subheader("🧠 Quiz")
                    # Display quizzes as bullet points
                    for quiz in quizzes:
                        quiz_list = quiz.split("\n")
                        for question in quiz_list:
                            if question.strip():
                                st.markdown(f"- {question.strip()}")

                except Exception as e:
                    st.error(f"Something went wrong: {e}")

        
