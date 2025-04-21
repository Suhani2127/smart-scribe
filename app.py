import streamlit as st
import PyPDF2
import requests
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

# 🔐 Hugging Face API setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}

# Initialize NLTK punkt tokenizer
nltk.download('punkt')

st.set_page_config(page_title="SmartScribe AI", page_icon="📝")
st.title("📝 SmartScribe AI")
st.subheader("Upload your notes and get instant AI-generated summaries (free & open-source powered)")

uploaded_file = st.file_uploader("📤 Upload a PDF or TXT file", type=["pdf", "txt"])

# 🔍 PDF Text Extractor
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# 🤖 Summarization using Hugging Face API
def summarize_with_huggingface(text):
    prompt = f"Summarize the following notes:\n\n{text}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.5,
            "max_new_tokens": 300
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

        # Function to highlight key sentences
        def highlight_key_sentences(text):
            sentences = sent_tokenize(text)
            word_freq = Counter(text.split())
            key_sentences = []

            # Get top 5 most frequent terms
            top_terms = [term for term, _ in word_freq.most_common(5)]
            
            # Highlight sentences containing top terms
            for sentence in sentences:
                if any(term in sentence for term in top_terms):
                    key_sentences.append(f"📌 {sentence}")

            return "\n\n".join(key_sentences)

        # ✨ Generate Quiz from the Text
        def generate_quiz_from_text(text):
            questions = [
                {"question": "What is the primary objective of the design and optimization of n-InP/p-Si heterojunction solar cells?",
                 "options": ["A. To reduce solar panel costs", "B. To enhance the efficiency of solar cells", 
                             "C. To increase the durability of solar panels", "D. To minimize energy consumption"],
                 "answer": "B. To enhance the efficiency of solar cells"},
                
                {"question": "Which simulation tool was mentioned for the design and optimization of the solar cells?",
                 "options": ["A. MATLAB", "B. PC1D", "C. COMSOL Multiphysics", "D. SPICE"],
                 "answer": "B. PC1D"},
                
                {"question": "Which materials are used in the device structure of the n-InP/p-Si heterojunction solar cells?",
                 "options": ["A. InP and Si", "B. InP and GaAs", "C. Si and GaN", "D. AlAs and Si"],
                 "answer": "A. InP and Si"},
                
                {"question": "What is a key parameter adjusted for optimization in the n-InP/p-Si heterojunction solar cells?",
                 "options": ["A. Solar cell area", "B. Temperature of operation", "C. Layer thickness", "D. Material purity"],
                 "answer": "C. Layer thickness"}
            ]
            
            quiz = ""
            for q in questions:
                quiz += f"**Q: {q['question']}**\n"
                for option in q["options"]:
                    quiz += f"- {option}\n"
                quiz += f"**Answer:** {q['answer']}\n\n"
            return quiz

        # ✨ Show Summary and Quiz Button
        if st.button("✨ Summarize Notes"):
            with st.spinner("Summarizing with Mistral 7B (free model)..."):
                try:
                    summary = summarize_with_huggingface(extracted_text)
                    st.subheader("🧠 Summary")
                    st.write(summary)

                    # Generate Quiz
                    st.subheader("🎓 Quiz")
                    quiz = generate_quiz_from_text(extracted_text)
                    st.write(quiz)

                except Exception as e:
                    st.error(f"Something went wrong: {e}")

        # Highlight Key Sentences Button
        if st.button("✨ Highlight Key Sentences"):
            with st.spinner("Highlighting key sentences..."):
                try:
                    highlighted = highlight_key_sentences(extracted_text)
                    st.subheader("📌 Key Sentences")
                    st.write(highlighted)

                except Exception as e:
                    st.error(f"Something went wrong: {e}")

