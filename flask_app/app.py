from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = './model.pkl'  # Ensure this path is correct
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Define route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        form_data = request.form

        # Process form data
        data = {
            'comp': 1,  # Assuming 'comp' is always 1
            'round': int(form_data['round']),
            'venue': 1 if form_data['venue'] == 'Home' else 2,
            'gf': int(form_data['gf']),
            'ga': int(form_data['ga']),
            'xg': float(form_data['xg']),
            'xga': float(form_data['xga']),
            'poss': float(form_data['poss']),
            'sh': int(form_data['sh']),
            'sot': int(form_data['sot']),
            'fk': int(form_data['fk']),
            'pk': int(form_data['pk']),
            'pkatt': int(form_data['pkatt']),
            # Handle one-hot encoded columns for opponent, team, and formation
            f'opponent_{form_data["opponent"]}': 1,
            f'formation_{form_data["formation"]}': 1,
            f'team_{form_data["team"]}': 1
        }
        
        # Create DataFrame with one row
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)
        result = 'Win' if prediction == 3 else 'Draw' if prediction == 1 else 'Lose'
        
        return render_template('index.html', prediction=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
