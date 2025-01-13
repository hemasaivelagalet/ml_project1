# Sentiment Analysis Web App

This project is a simple web application that performs sentiment analysis on movie reviews. The app classifies user-provided reviews as either **Positive** or **Negative** using a Logistic Regression model.

## Features

- Analyze movie reviews and predict sentiment (positive/negative).
- User-friendly interface powered by **Flask**.
- Pre-trained model using sample reviews for demonstration.

## Screenshots!

[Screenshot 2025-01-13 232227](https://github.com/user-attachments/assets/9450d445-23bc-4cf3-926d-d9ffc4ba58d7)

## Installation

Follow these steps to run the app locally:

### Prerequisites

- Python 3.7+
- `pip` (Python package manager)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-web-app.git
   cd sentiment-analysis-web-app



Install dependencies:

pip install -r requirements.txt

Run the app:

    python app.py

    Open the app in your browser: Navigate to http://127.0.0.1:5000/.

File Structure

sentiment-analysis-web-app/
│
├── app.py                # Main Flask application
├── requirements.txt      # List of Python dependencies
├── templates/
│   └── index.html        # HTML template for the web app
└── README.md             # Project documentation

Example Usage

    Enter a movie review in the input box.
    Click the Analyze Sentiment button.
    The app will display the sentiment of the review (Positive/Negative).

Model and Data

    The app uses a Logistic Regression model trained on a small set of manually labeled reviews.
    Feature extraction is performed using TF-IDF Vectorization.

Technologies Used

    Flask: Backend web framework.
    scikit-learn: For machine learning model training and evaluation.
    HTML/CSS: For the frontend user interface.

Future Enhancements

    Add support for more extensive datasets for better model accuracy.
    Include additional machine learning models (e.g., SVM, neural networks).
    Deploy the app on cloud platforms like Heroku or AWS.

Contributing

Contributions are welcome! If you’d like to improve the app, please fork the repository and submit a pull request.
License

This project is licensed under the MIT License. See the LICENSE file for details.
Contact

For questions or feedback, feel free to reach out:

    Email: your-email@example.com
    GitHub: your-username


### Additional Steps:
1. Replace placeholders like `your-username`, `your-email@example.com`, and `screenshot.png` with your actual details.
2. Add a `requirements.txt` file with the necessary dependencies:
   ```plaintext
   flask
   numpy
   scikit-learn

    Include a screenshot of your app (screenshot.png) in the project directory for visual reference.

Feel free to ask if you need further customization!
