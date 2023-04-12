import pandas as pd
import numpy as np
import re

# Load the dataset
url = 'https://raw.githubusercontent.com/ShresthaSudip/SMS_Spam_Detection_DNN_LSTM_BiLSTM/master/SMSSpamCollection'
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Data cleaning and preprocessing
df['message'] = df['message'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x.lower()))

# Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a binary classification model using Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Test the model and evaluate the performance
y_pred = model.predict(X_test_vec)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Ask the user for input messages to predict whether they are spam or not
while True:
    user_input = input("Enter a message to predict whether it is spam or not (enter 'exit' to quit): ")
    if user_input == 'exit':
        break
    else:
        user_input_cleaned = re.sub('[^a-zA-Z0-9\s]', '', user_input.lower())
        user_input_vec = vectorizer.transform([user_input_cleaned])
        prediction = model.predict(user_input_vec)[0]
        if prediction == 'ham':
            print("Prediction: Not spam")
        else:
            print("Prediction: Spam")

