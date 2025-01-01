from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("credit_card_fraud_model.pkl")

@app.route('/')
def home():
    return "Welcome to the Credit Card Fraud Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data sent via POST request
    data = request.get_json()

    # Ensure the data is in the correct format
    try:
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # Return the prediction result
        if prediction == 0:
            result = "Normal Transaction"
        else:
            result = "Fraudulent Transaction"
        
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
 
