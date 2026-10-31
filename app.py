from os import path
from flask import Flask, render_template, request
import joblib
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent     # Get the folder where app.py is located

model, vectorizer = joblib.load(BASE_DIR / 'sms_spam_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():

  # Set default values
  prediction = None
  message = ""

  # Checks if user submitted a message
  if request.method == 'POST':
    message = request.form.get('message','').strip()   # Gets the SMS entered by the user

    # Predicts only when message is not empty
    if message:
      features = vectorizer.transform([message])      # Convert SMS into numerical features
      result = model.predict(features)[0]             # Predict class 0 or 1
      prediction = 'Spam' if result == 1 else 'Ham'

  return render_template(
     'index.html',
     prediction=prediction,
     message=message
 )



if __name__ == '__main__':
  app.run(debug=True)