# Email and SMS Spam Detection

## Overview
This project focuses on detecting spam messages in emails and SMS using machine learning techniques. By analyzing text features and employing a variety of classification algorithms, the model identifies whether a given message is spam or not. The application also includes a user-friendly interface to facilitate easy and intuitive interaction for spam detection.

📍 **App link:** https://spam-detection-n53dj6pcmkhwnx72wykdtl.streamlit.app/

## Features
- **Text Preprocessing**: Cleans and preprocesses the text data by removing stopwords, punctuation, and applying stemming.
- **Model Training**: Utilizes various machine learning algorithms like Multinomial Naive Bayes, Support Vector Classifier, Random Forest, Extra Trees Classifier, and XGBoost to predict whether a message is spam.
- **User Interface**: A web-based application built with Streamlit to allow users to input text and get instant predictions.

## Requirements
- Python 3.x
- **Libraries**: pandas, numpy, scikit-learn, nltk, streamlit, matplotlib, seaborn, wordcloud, xgboost, pickle.

## Files
- **Spam_Detection.ipynb:** Contains the code for data preprocessing, model training, and evaluation.
- **spam.csv:** Dataset containing labeled SMS and email messages for training and evaluation.
- **vectorizer.pkl:** Pickle file of the fitted TF-IDF vectorizer.
- **model.pkl:** Pickle file of the trained machine learning model.
- **app.py:** Streamlit application script to host the spam detection web app.
