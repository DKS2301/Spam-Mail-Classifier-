import nltk
import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Ensure stopwords are downloaded
nltk.download("stopwords")

# Properly load stopwords
stop_words = set(stopwords.words("english"))

ps = PorterStemmer()

# Loading the trained model
with open("./model/spam-classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Loading the vectorizer
with open("./model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def transform_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = nltk.word_tokenize(text)  # Tokenize
    words = [ps.stem(word) for word in words if word not in stop_words]  # Remove stopwords & stem
    return " ".join(words)

# Function to predict spam or ham
def predict_spam(message):
    message = transform_text(message)
    transformed_message = vectorizer.transform([message])
    prediction = model.predict(transformed_message)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Streamlit UI
st.title("📩 Spam SMS Classifier")
st.write("Enter an SMS message below to check if it's spam or not.")

# Input text box
message = st.text_area("Enter SMS:", "")

# Predict button
if st.button("Classify"):
    if message.strip():
        result = predict_spam(message)
        st.success(f"Prediction: **{result}**")
    else:
        st.warning("Please enter a message.")

# Footer
st.markdown("---")
st.write("Built with ❤️ using Streamlit and Machine Learning")
