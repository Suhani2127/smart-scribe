import streamlit as st
import PyPDF2

#import requests

# Hugging Face API setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
headers = {
    "Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"
}

def generate_text(prompt):
    payload = {"inputs": prompt}
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "Error: Unable to fetch generated text."

# Example usage
st.title("SmartScribe AI")
input_prompt = st.text_area("Enter a prompt to generate text:")
if input_prompt:
    generated_text = generate_text(input_prompt)
    st.write("Generated Text:", generated_text)


headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}



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

        # ✨ Summarize Button
        if st.button("✨ Summarize Notes"):
            with st.spinner("Summarizing with Mistral 7B (free model)..."):
                try:
                    summary = summarize_with_huggingface(extracted_text)
                    st.subheader("🧠 Summary")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")


