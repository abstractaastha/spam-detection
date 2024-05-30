import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Function to preprocess and clean the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)

    cleaned_text = []
    for token in text:
        if token.isalnum():
            cleaned_text.append(token)

    text = cleaned_text[:]
    cleaned_text.clear()

    for token in text:
        if token not in stopwords.words('english') and token not in string.punctuation:
            cleaned_text.append(token)

    text = cleaned_text[:]
    cleaned_text.clear()

    for token in text:
        cleaned_text.append(ps.stem(token))

    return " ".join(cleaned_text)

# Load the vectorizer and model from the pickle files
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 24px;
    }
    .stTextArea>div>textarea {
        font-size: 16px;
    }
    .header {
        font-size: 30px;
        font-weight: bold;
        color: #333333;
        text-align: center;
        margin-bottom: 20px;
    }
    .footer {
        font-size: 14px;
        color: #777777;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title with custom styling
st.markdown('<div class="header">SMS & Email Spam Detector</div>', unsafe_allow_html=True)

# Add description
st.write("""
### Welcome to the Spam Detection App
Use this tool to classify SMS messages and emails as **Spam** or **Not Spam**.
Simply enter the message text below and click on **Predict** to see the result.
""")

# Text area for user input
user_input = st.text_area("Enter the text of the message or email")

# Predict button
if st.button('Predict'):
    # Preprocess the input text
    processed_text = preprocess_text(user_input)
    # Vectorize the processed text
    vectorized_input = vectorizer.transform([processed_text])
    # Make a prediction using the loaded model
    prediction = model.predict(vectorized_input)[0]
    # Display the prediction
    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

# Footer
st.markdown('<div class="footer">Created with ❤ by Aastha</div>', unsafe_allow_html=True)
