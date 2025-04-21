import streamlit as st
import PyPDF2
import requests
import textwrap

# 🔐 Hugging Face API setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}

st.set_page_config(page_title="SmartScribe AI", page_icon="📝")
st.title("📝 SmartScribe AI")
st.subheader("Upload your notes and get instant AI-generated summaries and quizzes")

uploaded_file = st.file_uploader("📤 Upload a PDF or TXT file", type=["pdf", "txt"])

# 🔍 PDF Text Extractor
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# 🧠 Summarization and Quiz Generation using Hugging Face API
def summarize_with_huggingface(text):
    prompt = f"Summarize the following notes:\n\n{text}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.5,
            "max_new_tokens": 150  # reduced token size for faster processing
        }
    }

    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# 📝 Quiz Generation using Hugging Face API
def generate_quiz(text):
    prompt = f"Create a quiz with multiple-choice questions based on the following notes:\n\n{text}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.5,
            "max_new_tokens": 150  # reduced token size for faster processing
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

        # ✨ Split text into smaller chunks for faster processing
        chunk_size = 500  # limit to 500 tokens per chunk
        text_chunks = textwrap.wrap(extracted_text, chunk_size)

        # ✨ Summarize Button
        if st.button("✨ Summarize Notes"):
            with st.spinner("Summarizing..."):
                summaries = []
                for chunk in text_chunks:
                    try:
                        summary = summarize_with_huggingface(chunk)
                        summaries.append(summary)
                    except Exception as e:
                        st.error(f"Error while summarizing: {e}")
                
                st.subheader("🧠 Summary")
                st.write("\n".join(summaries))

        # ✨ Quiz Button
        if st.button("📝 Generate Quiz"):
            with st.spinner("Generating quiz..."):
                quizzes = []
                for chunk in text_chunks:
                    try:
                        quiz = generate_quiz(chunk)
                        quizzes.append(quiz)
                    except Exception as e:
                        st.error(f"Error while generating quiz: {e}")

                st.subheader("🎓 Quiz")
                st.write("\n".join(quizzes))
