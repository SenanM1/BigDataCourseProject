from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import streamlit as st
import cleantext
import re
import nltk
from nltk.corpus import stopwords
import plotly.express as px
from joblib import load

# Download the VADER lexicon
nltk.download('vader_lexicon')

st.header('Sentiment Analysis')

# Load the Naive Bayes model
naive_bayes_model = load('naive_bayes_model.joblib')
vectorizer = load('vectorizer.joblib')

# Load Azerbaijani stopwords
azerbaijani_stopwords = stopwords.words('azerbaijani')

def clean_text(text, stopwords=azerbaijani_stopwords):
    # Check if the input is a string
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = ' '.join(word for word in text.split() if word not in stopwords)
    else:
        # If not a string, return an empty string
        text = ''
    return text

# Function for TextBlob analysis
def textblob_analysis(text):
    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 2), round(blob.sentiment.subjectivity, 2)

# Function for VADER analysis
def vader_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    compound_score = analyzer.polarity_scores(text)['compound']
    return compound_score

# Define the analyze function
def analyze(x):
    if x >= 0.5:
        return 'Positive'
    elif x <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

# Initialize df outside the CSV analysis section
df = None

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        # Add a dropdown to select the analysis model
        selected_model = st.selectbox('Select Sentiment Analysis Model:', ['TextBlob', 'VADER'])
        
        if selected_model == 'TextBlob':
            polarity, subjectivity = textblob_analysis(text)
            st.write('Polarity (TextBlob): ', polarity)
            st.write('Subjectivity (TextBlob): ', subjectivity)
        elif selected_model == 'VADER':
            compound_score = vader_analysis(text)
            st.write('Compound Score (VADER): ', compound_score)

with st.expander('Azerbaijani Sentiment Analysis'):
    azerbaijani_text = st.text_input('Enter Azerbaijani text: ')

    if azerbaijani_text:
        # Preprocess the text
        cleaned_text = clean_text(azerbaijani_text)  # Use the same clean_text function you have in your notebook
        transformed_text = vectorizer.transform([cleaned_text])  # Use the same vectorizer object you saved in your notebook

        # Predict sentiment
        prediction = naive_bayes_model.predict(transformed_text)
        st.write('Sentiment: ', prediction[0])


pre = st.text_input('Clean Text: ')
if pre:
    st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True,
                             stopwords=True, lowercase=True, numbers=True, punct=True))

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    if upl:
        df = pd.read_csv(upl)  # Read CSV

        # Select the language/model for analysis
        analysis_language = st.selectbox('Select Language for Sentiment Analysis:', ['English', 'Azerbaijani'])

        if analysis_language == 'Azerbaijani':
            # Azerbaijani sentiment analysis using Naive Bayes
            text_column = st.selectbox('Select the column for Azerbaijani text analysis:', df.columns)

            # Load your Naive Bayes model and vectorizer here
            naive_bayes_model = load('naive_bayes_model.joblib')
            vectorizer = load('vectorizer.joblib')

            def azerbaijani_sentiment_analysis(text):
                cleaned_text = clean_text(text)
                transformed_text = vectorizer.transform([cleaned_text])
                prediction = naive_bayes_model.predict(transformed_text)
                return prediction[0]

            df['Azerbaijani Sentiment'] = df[text_column].apply(azerbaijani_sentiment_analysis)
            st.write(df)
        else:
            df = pd.read_csv(upl)  # Read CSV instead of Excel

            # Allow the user to select the column containing text data
            text_column = st.selectbox('Select the column for analysis:', df.columns)

            # Allow the user to choose columns to keep
            columns_to_keep = st.multiselect('Select columns to keep:', df.columns, default=[text_column])

            # Keep selected columns
            df = df[columns_to_keep]

            # Add a dropdown to select the analysis model
            selected_model = st.selectbox('Select Sentiment Analysis Model:', ['TextBlob', 'VADER'])

            if selected_model == 'TextBlob':
                @st.cache_data
                def score(x):
                    blob = TextBlob(x)
                    return blob.sentiment.polarity
            elif selected_model == 'VADER':
                @st.cache_data
                def score(x):
                    analyzer = SentimentIntensityAnalyzer()
                    return analyzer.polarity_scores(x)['compound']

            df['score'] = df[text_column].apply(score)

            # Add analysis based on the selected model
            if selected_model == 'TextBlob':
                df['analysis'] = df['score'].apply(analyze)
            elif selected_model == 'VADER':
                df['analysis'] = df['score'].apply(lambda x: 'Positive' if x >= 0.5 else 'Negative' if x <= -0.5 else 'Neutral')

            # Display all rows
            st.write(df)

            # Download button for the processed DataFrame
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )

# Comparison section
with st.expander('Comparison'):
    if df is not None and 'score' in df.columns and 'analysis' in df.columns:
        st.write('Comparison of Sentiment Analysis Models:')
        
        textblob_avg_polarity = df['score'].mean()
        vader_avg_polarity = df['score'].mean()

        st.write('Average Polarity (TextBlob): ', textblob_avg_polarity)
        st.write('Average Polarity (VADER): ', vader_avg_polarity)

        # Visualization - Bar Chart
        fig = px.bar(
            x=['TextBlob', 'VADER'],
            y=[textblob_avg_polarity, vader_avg_polarity],
            labels={'x': 'Sentiment Analysis Model', 'y': 'Average Polarity'},
            title='Average Polarity Comparison'
        )
        st.plotly_chart(fig)
