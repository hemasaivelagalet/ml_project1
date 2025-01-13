from flask import Flask, request, jsonify, render_template
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Initialize Flask App
app = Flask(__name__)

# Define stopwords manually
stop_words = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"<[^>]*>", "", text)  # Remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Training the model
reviews = [
    "The movie was fantastic! I really enjoyed it.",
    "This was a terrible movie. I wasted my time.",
    "Absolutely loved it. Would watch again!",
    "Worst movie ever. Do not recommend.",
    "It was an okay movie. Nothing special."
]
labels = [1, 0, 1, 0, 0]  # 1 = Positive, 0 = Negative
processed_reviews = [preprocess_text(review) for review in reviews]
tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=1, max_df=0.9)
X = tfidf_vectorizer.fit_transform(processed_reviews).toarray()
y = np.array(labels)

model = LogisticRegression()
model.fit(X, y)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['review']
    processed_text = preprocess_text(input_text)
    input_features = tfidf_vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(input_features)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return render_template('index.html', review=input_text, prediction=sentiment)

# Run the App
if __name__ == '__main__':
    app.run(debug=True)
