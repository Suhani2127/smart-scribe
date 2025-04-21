import streamlit as st
import PyPDF2
import openai

# Load API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("📚 SmartScribe AI - Step 2")
st.subheader("Upload your notes (PDF or TXT) and get instant summaries!")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

extracted_text = ""

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    
    if file_type == "pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "txt":
        extracted_text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type.")

    st.success("✅ Text extracted successfully!")
    with st.expander("🔍 Show Extracted Text"):
        st.write(extracted_text)

    if extracted_text:
        if st.button("✨ Summarize Notes"):
            with st.spinner("Summarizing with GPT..."):
                try:
                    from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that summarizes class notes."},
        {"role": "user", "content": f"Summarize these notes:\n\n{extracted_text}"}
    ],
    temperature=0.5,
    max_tokens=500
)

summary = response.choices[0].message.content

                    st.subheader("📝 Summary")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")
