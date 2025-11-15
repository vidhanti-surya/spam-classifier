from flask import Flask, render_template, request
import joblib
import os
import sys

# Initialize Flask app
app = Flask(__name__)

# Define file paths
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

# --- Load model and vectorizer with error handling ---
try:
    # Check if files exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Missing one or both files: {MODEL_PATH} and {VECTORIZER_PATH}. "
                                "Ensure the files contain the *fitted* model and vectorizer.")

    # Load the fitted model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and Vectorizer loaded successfully.")

except FileNotFoundError as e:
    # Handle missing files or the case where joblib.load fails due to an unfitted object
    print(f"❌ CRITICAL ERROR: {e}", file=sys.stderr)
    print("Please ensure you have trained your model and saved the *fitted* objects to the correct paths.", file=sys.stderr)
    # Exiting or using a dummy model is safer than proceeding with an unfitted object
    sys.exit(1)
except Exception as e:
    # Handle other loading issues (e.g., corrupted file, wrong object type)
    print(f"❌ An unexpected error occurred during model loading: {e}", file=sys.stderr)
    sys.exit(1)
# ----------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        
        # 1. Transform the input using the loaded vectorizer
        vector_input = vectorizer.transform([message])
        
        # 2. Predict using the loaded, fitted model
        # If the code reaches here, it assumes the 'model' is fitted.
        # If the NotFittedError still occurs here, it means the content of 'model.pkl'
        # is an unfitted estimator, and you must correct the file generation process.
        try:
            result = model.predict(vector_input)[0]
            output = "Spam" if result == 1 else "Not Spam"
            return render_template('index.html', prediction_text=f"Result: {output}")
        except Exception as e:
            # Catch any other runtime error during prediction
            return render_template('index.html', prediction_text=f"Error during prediction: {e}")

if __name__ == "__main__":
    # Remove debug=True for production
    app.run(debug=True)