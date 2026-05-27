# NLP-Project

A comprehensive Natural Language Processing project focused on **fake news detection** using machine learning and NLP techniques.

## 📋 Overview

This project implements a machine learning-based solution to detect and classify news articles as real or fake. It combines TF-IDF vectorization with sentiment analysis and linguistic features to create a robust classification model.

## 🎯 Project Components

### Main Files

- **`NLP Script.ipynb`** - Jupyter notebook containing the main analysis, model training, and evaluation workflow
- **`Fake news detection.py`** - Streamlit web application for interactive fake news detection
- **`fake_real_news.csv`** - Dataset containing labeled news articles (real and fake)
- **`fake_news_model.pkl`** - Pre-trained machine learning model (serialized)
- **`tfidf_vectorizer.pkl`** - TF-IDF vectorizer for text feature extraction

## 🚀 Features

### Text Processing
- **Normalization**: Removes special characters, converts to lowercase
- **Tokenization**: Splits text into individual tokens using NLTK
- **Lemmatization**: Reduces words to their base forms
- **Stop Words Removal**: Eliminates common English stop words

### Feature Engineering
- **Word Count**: Number of words in the text
- **Exclamation Marks**: Count of exclamation marks (sensationalism indicator)
- **Positive Word Count**: Uses NLTK opinion lexicon
- **Negative Word Count**: Uses NLTK opinion lexicon
- **TF-IDF Features**: Vectorized text representation

### Sentiment Analysis
- Visual representation of positive vs. negative words in articles
- Sentiment distribution charts using Plotly

## 📦 Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

**Dependencies:**
- pandas
- plotly
- streamlit
- nltk
- joblib
- scikit-learn

## 🏃 How to Use

### 1. Interactive Web Application
Run the Streamlit app to analyze articles in real-time:
```bash
streamlit run "Fake news detection.py"
```

Then:
1. Enter the article title in the sidebar
2. Enter the article content
3. Click "Analyze" to get the prediction
4. View sentiment analysis visualization

### 2. Jupyter Notebook
Open and run `NLP Script.ipynb` to:
- Explore the dataset
- Train and evaluate the model
- Generate visualizations and metrics
- Experiment with different preprocessing techniques

## 🔍 How It Works

1. **Input Processing**: Article title and content are collected from the user
2. **Text Preprocessing**: 
   - Normalizes text
   - Removes stop words
   - Applies lemmatization
3. **Feature Extraction**:
   - Calculates linguistic features (word count, punctuation, sentiment)
   - Converts text to TF-IDF vectors
4. **Prediction**: The trained model classifies the article as real (✅) or fake (⚠️)
5. **Visualization**: Displays sentiment analysis with interactive charts

## 📊 Model Details

The model uses:
- **Algorithm**: Scikit-learn classifier (details in notebook)
- **Training Data**: `fake_real_news.csv` dataset
- **Feature Set**: TF-IDF + linguistic features
- **Preprocessing**: Complete text normalization pipeline

## 👥 Authors

Created by:
- Johnny Chreim
- Benjamin Jacobsen
- Rohan Taneja

## 📝 Dataset

The project uses the `fake_real_news.csv` dataset containing labeled news articles. The dataset includes:
- Article titles
- Article content
- Labels (real/fake)

## 🛠️ Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/Johnny386/NLP-Project.git
cd NLP-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK resources (automatically done on first run of the app):
```python
import nltk
nltk.download('opinion_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

4. Run the application:
```bash
streamlit run "Fake news detection.py"
```

## 📈 Performance & Results

See the Jupyter notebook for:
- Model accuracy and metrics
- Confusion matrices
- Feature importance analysis
- ROC curves and AUC scores

## 🔗 Technologies Used

- **Python 3.x**
- **NLTK** - Natural Language Processing
- **Scikit-learn** - Machine Learning
- **Pandas** - Data Processing
- **Plotly** - Data Visualization
- **Streamlit** - Web Application Framework
- **Joblib** - Model Serialization

## 📄 License

This project is open source and available on GitHub.

---

**Status**: Active  
**Last Updated**: April 2025
