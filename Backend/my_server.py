import numpy as np
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
data = pd.read_excel(r"C:\Users\alhar\Downloads\online+retail\Online Retail.xlsx")
app = Flask(__name__)

# Load the trained LSTM model
model = load_model('unitprice_prediction_model_lstm (1).h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded successfully!")

# Load the OneHotEncoder
with open('ohe_country.pkl', 'rb') as f:
    ohe_country = pickle.load(f)
print("OneHotEncoder loaded successfully!")

# Load additional parameters
with open('model_params.pkl', 'rb') as f:
    params = pickle.load(f)
max_sequence_length = params['max_sequence_length']
epsilon = params['epsilon']

# Get the list of countries from the OneHotEncoder categories
countries_list = ohe_country.categories_[0].tolist()
# API endpoint to get distinct countries from the dataset
@app.route('/api/countries', methods=['GET'])
def get_countries():
    # Extract distinct values from the 'Country' column
    distinct_countries = data['Country'].dropna().unique().tolist()

    # Create an array of objects with 'name' and 'id' for Appsmith Select component
    countries_list = [{"name": country, "id": index} for index, country in enumerate(distinct_countries)]

    # Return as JSON
    return jsonify(countries_list)
# API endpoint for making predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get 'country' and 'description' from the POST request
        req_data = request.get_json()
        description = req_data.get('description')
        country = req_data.get('country')

        if description is None or country is None:
            return jsonify({"error": "Please provide both 'description' and 'country' in the request body."}), 400

        # Check if the input country exists in the list of countries from training
        if country not in countries_list:
            return jsonify({"error": f"Country '{country}' not found in training data."}), 400

        # Preprocess the description (tokenize and pad)
        description_seq = tokenizer.texts_to_sequences([description.lower()])
        description_padded = pad_sequences(description_seq, maxlen=max_sequence_length)

        # Preprocess the country (One-Hot encode)
        country_encoded = ohe_country.transform([[country]])

        # Pass the preprocessed inputs to the model for prediction
        prediction_log = model.predict([description_padded, country_encoded])

        # Inverse log transformation to get the actual price
        prediction = np.exp(prediction_log[0][0]) - epsilon

        # Ensure no negative prediction
        prediction = max(prediction, 0)

        # Return the predicted price
        return jsonify({"predicted_unit_price": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
