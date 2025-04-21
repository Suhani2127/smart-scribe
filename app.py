import streamlit as st
import PyPDF2
import requests

# 🔐 Hugging Face API setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
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
            with st.spinner("Summarizing..."):
                try:
                    summary = summarize_with_huggingface(extracted_text)
                    st.subheader("🧠 Summary")
                    st.write(summary)

                    # Flashcards and Quiz Generation
                    flashcards = generate_flashcards(summary)
                    if not flashcards:  # Check if flashcards are empty
                        st.warning("⚠️ No flashcards generated. The summary might be too short or unclear.")
                    else:
                        quiz = generate_quiz_from_flashcards(flashcards)

                        # Display Flashcards
                        st.subheader("💡 Key Points & Flashcards")
                        for card in flashcards:
                            st.write(f"**Q: {card['question']}**")
                            st.write(f"**A: {card['answer']}**")

                        # Display Quiz
                        st.subheader("📝 Quiz")
                        for question in quiz:
                            st.write(f"**Q: {question['question']}**")
                            choices = question["choices"]
                            user_answer = st.radio(f"Select an answer", choices, key=question['question'])
                            if user_answer == question["correct_answer"]:
                                st.success("✅ Correct!")
                            else:
                                st.error("❌ Incorrect")

                except Exception as e:
                    st.error(f"Something went wrong: {e}")


# Function to generate flashcards from the summary
def generate_flashcards(summary):
    # This is where you can parse out key points from the summary
    flashcards = []

    if not summary:
        return flashcards  # Return empty list if no summary is provided

    # Example structure for the flashcards:
    flashcards.append({"question": "What is the main idea of the text?", "answer": summary[:100]})  # First 100 characters as an example
    flashcards.append({"question": "What is the purpose of this document?", "answer": summary[100:200]})

    # You can expand this as needed to focus on critical points
    return flashcards


# Function to generate quiz from flashcards
def generate_quiz_from_flashcards(flashcards):
    quiz = []
    if not flashcards:
        return quiz  # Return empty list if no flashcards are available
    for card in flashcards:
        question, answer = card["question"], card["answer"]
        quiz.append({
            "question": question,
            "choices": [answer, "Wrong answer 1", "Wrong answer 2", "Wrong answer 3"],
            "correct_answer": answer
        })
    return quiz

