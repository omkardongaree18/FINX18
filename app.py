from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
with open('investment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset for company recommendations
df = pd.read_csv("bank_investment_dataset_updated.csv")

@app.route('/')
def home():
    return "Investment Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        features = np.array([list(data.values())]).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]

        response = {"investment_success": bool(prediction)}

        if prediction == 1:
            # Find the best companies based on past successful investments
            successful_investments = df[df["Successful_Investment"] == 1].copy()
            distances = np.linalg.norm(successful_investments.drop(columns=["Successful_Investment", "Failed_Investment", "Company_Name"]).values - features, axis=1)
            top_10_indices = np.argsort(distances)[:10]
            top_10_companies = successful_investments.iloc[top_10_indices]["Company_Name"].tolist()
            
            response["recommended_companies"] = top_10_companies
        else:
            response["message"] = "Investment might not be successful. Consider adjusting parameters."

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
