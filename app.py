import streamlit as st
import PyPDF2
import requests
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import textstat
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import sent_tokenize

# 🔐 Hugging Face API setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="SmartScribe AI", page_icon="📝")
st.title("📝 SmartScribe AI")
st.subheader("Upload your notes and get instant AI-generated flashcards (free & open-source powered)")

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

# 🤖 Generate Flashcards using Hugging Face API
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

# Word Cloud visualization
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot()

# Sentiment Analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        sentiment_label = "Positive"
    elif sentiment < 0:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    return sentiment_label

# Readability Score
def get_readability_score(text):
    score = textstat.flesch_kincaid_grade(text)
    return score

# Named Entity Recognition (NER)
def extract_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# Keyword Extraction (TF-IDF)
def extract_keywords(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = tfidf_matrix.sum(axis=0).argsort()[0, ::-1]
    top_keywords = [feature_names[i] for i in sorted_indices[:10]]  # Top 10 keywords
    return top_keywords

# Topic Modeling (LDA)
def extract_topics(text, num_topics=3):
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    topics = lda.components_
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = []
    for topic in topics:
        top_keywords = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topic_keywords.append(top_keywords)
    return topic_keywords

# Bullet Point Summary
def summarize_bullet_points(text):
    sentences = sent_tokenize(text)
    summary = []
    for sentence in sentences:
        if len(sentence.split()) > 6:  # Filter out very short sentences
            summary.append(f"- {sentence}")
    return summary


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

        # ✨ Flashcards Button
        if st.button("✨ Generate Flashcards"):
            with st.spinner("Generating flashcards..."):
                try:
                    # Split text into manageable chunks
                    chunks = split_text(extracted_text)

                    flashcards = []
                    for chunk in chunks:
                        flashcards_chunk = generate_flashcards_with_huggingface(chunk)
                        flashcards.append(flashcards_chunk)

                    st.subheader("🧠 Flashcards")
                    # Display flashcards as bullet points
                    for flashcard in flashcards:
                        flashcard_list = flashcard.split("\n")
                        for card in flashcard_list:
                            if card.strip():
                                st.markdown(f"- {card.strip()}")

                    # ✨ Analytics Section
                    st.subheader("📊 Text Analytics and Insights")

                    # Word Cloud
                    st.subheader("🔠 Word Cloud of the Document")
                    generate_word_cloud(extracted_text)

                    # Sentiment Analysis
                    st.subheader("🧠 Sentiment Analysis")
                    sentiment = analyze_sentiment(extracted_text)
                    st.write(f"The sentiment of the document is: {sentiment}")

                    # Readability Score
                    st.subheader("📊 Readability Score")
                    score = get_readability_score(extracted_text)
                    st.write(f"Readability score (Flesch-Kincaid Grade Level): {score}")

                    # Named Entity Recognition
                    st.subheader("🔑 Named Entity Recognition")
                    entities = extract_entities(extracted_text)
                    st.write("Identified Entities:")
                    for entity in entities:
                        st.write(f"{entity[0]} ({entity[1]})")

                    # Keyword Extraction (TF-IDF)
                    st.subheader("🔑 Key Keywords")
                    keywords = extract_keywords(extracted_text)
                    st.write(f"Top Keywords: {', '.join(keywords)}")

                    # Topic Modeling (LDA)
                    st.subheader("📝 Topic Modeling")
                    topics = extract_topics(extracted_text)
                    for i, topic in enumerate(topics, 1):
                        st.write(f"Topic {i}: {', '.join(topic)}")

                    # Bullet Point Summary
                    st.subheader("🔹 Bullet Point Summary")
                    bullet_points = summarize_bullet_points(extracted_text)
                    for point in bullet_points:
                        st.write(point)

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
