# Libraries
import pandas as pd
import plotly.express as px
import streamlit as st
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
import joblib
import re
import sklearn

# Download necessary NLTK resources
nltk.download('opinion_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')


# Initialize stopwords outside the functions
stop_words = nltk.corpus.stopwords.words('english')

# Function to count positive words
def Positive_words(text):
    pos_list = set(opinion_lexicon.positive())
    text = word_tokenize(text)
    return len([token for token in text if token in pos_list])

# Function to count negative words
def Negative_words(text):
    neg_list = set(opinion_lexicon.negative())
    text = word_tokenize(text)
    return len([token for token in text if token in neg_list])

# Function to generate variables
def generateVariables(df):
    nbwords = []
    exclamations = []
    Pos_count = []
    Neg_count = []

    for sentence in df['text']:
        nbwords.append(len(sentence.split()))
        exclamations.append(sentence.count('!'))
        Pos_count.append(Positive_words(sentence))
        Neg_count.append(Negative_words(sentence))

    df['WordCount'] = nbwords
    df['ExclCount'] = exclamations
    df['PosCount'] = Pos_count
    df['NegCount'] = Neg_count

    return df

# Function to normalize text
def normalize_document(doc):
    doc = re.sub(r"[^a-zA-Z\s]", '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = word_tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)

# Preprocess text function
def preprocess_text(text):
    df = pd.DataFrame([text], columns=['text'])
    df['text'] = df['text'].fillna('').astype(str)
    df_norm_titles = pd.DataFrame({"norm_title": [normalize_document(title) for title in df['text']]})
    df = df.join(df_norm_titles)

    X = generateVariables(df)
    
    # Load the vectorizer and transform the text
    vectorizer = joblib.load("./tfidf_vectorizer.pkl")
    X_tfidf = vectorizer.transform(X['norm_title'])  # Use transform (not fit_transform)
    
    return X_tfidf

# Streamlit App

# Sidebar for instructions and input
st.sidebar.title("Fake News Detection")
st.sidebar.write("This is a simple fake news detection app. Enter the article's title and content to check if the news is real or fake.")
st.sidebar.header("How to Use:")
st.sidebar.write("1. Enter the title and content of the article you want to analyze.")
st.sidebar.write("2. Click 'Analyze' to see the result.")

input_title = st.sidebar.text_area("Article Title:", "Type here...")
input_text = st.sidebar.text_area("Article Content:", "Type here...")

# Display the app title
st.title("Fake News Detection")
st.write("This app detects whether the news is real or fake based on the provided text.")

if st.sidebar.button("Analyze"):
    with st.spinner('Analyzing the article...'):
        try:
            # Load the trained model
            model = joblib.load("./fake_news_model.pkl")

            # Preprocess the input text
            preprocessed_text = preprocess_text(input_text)

            # Make a prediction
            prediction = model.predict(preprocessed_text)

            # Display prediction result
            if prediction == 1:
                st.error("The news is likely to be fake. ⚠️")
            else:
                st.success("The news is likely to be real. ✅")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Optional: Visualizations
st.subheader("Sentiment Analysis")
st.write("Below is a visual representation of positive and negative word counts in the article.")

# Plot sentiment analysis if needed
pos_count = Positive_words(input_text)
neg_count = Negative_words(input_text)

# Create a bar chart for positive vs. negative words
fig = px.bar(x=["Positive", "Negative"], y=[pos_count, neg_count], labels={"x": "Sentiment", "y": "Count"})
st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("Created by Johnny Chreim, Benjamin Jacobsen, Rohan Taneja")
st.markdown("This app is built using Streamlit and various NLP libraries.")
