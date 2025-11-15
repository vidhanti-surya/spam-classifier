# app.py

from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load fitted model and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vector_input = cv.transform(data)
        result = model.predict(vector_input)[0]

        output = "🚫 Spam" if result == 1 else "✅ Not Spam"
        return render_template('index.html', prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)
