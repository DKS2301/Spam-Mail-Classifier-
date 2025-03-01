# Spam Mail Classifier

This project is a machine learning-based spam classifier that predicts whether an SMS message is spam or not spam using Natural Language Processing (NLP) and Machine Learning (ML).

## Features

- Uses TF-IDF Vectorization for text preprocessing.
- Machine Learning Model trained on a spam dataset.
- Streamlit Web App for easy classification.
- Simple and user-friendly UI.

## Project Structure

```
Spam-Mail-Classifier/
│── model/                  # Saved model & vectorizer
│   ├── spam-classifier.pkl  # Trained ML model
│   ├── vectorizer.pkl       # TF-IDF Vectorizer
│── app.py                   # Streamlit Web App
│── requirements.txt         # Dependencies
│── README.md                # Documentation
```

## Installation & Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/DKS2301/Spam-Mail-Classifier-.git
   cd Spam-Mail-Classifier-
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## How It Works

1. Preprocessing: The text is tokenized, stopwords are removed, and words are stemmed.
2. Vectorization: Converts text into numerical features using TF-IDF.
3. Model Prediction: Uses a trained machine learning model to classify the message.

## Contributing

Feel free to fork this repository and submit pull requests.

## License

This project is licensed under the MIT License.

Built using Python, Scikit-Learn, and Streamlit.
