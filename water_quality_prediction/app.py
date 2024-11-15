from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('best_model.sav')  # Correct model file
    scaler = joblib.load('scaler.joblib')  # Ensure the scaler is also loaded
    print("Model and Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Retrieve form inputs and convert to float
        temperature = float(request.form['temperature'])
        ph = float(request.form['ph'])
        nitrate = float(request.form['nitrate'])
        dissolved_oxygen = float(request.form['dissolved_oxygen'])
        salinity = float(request.form['salinity'])

        print(f"Received inputs: {temperature}, {ph}, {nitrate}, {dissolved_oxygen}, {salinity}")

        # Prepare input data for prediction
        input_data = np.array([[temperature, ph, nitrate, dissolved_oxygen, salinity]])

        # Scale the input data using the saved scaler
        input_data_scaled = scaler.transform(input_data)

        # Make the prediction using the model
        prediction = model.predict(input_data_scaled)[0]
        
        print(f"Prediction: {prediction}")

        # Render the result page with the prediction
        return render_template('result.html', prediction=prediction)
    
    except ValueError as ve:
        print(f"Input value error: {ve}")
        return "Error: Please check the input values. They should be numeric and valid."
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "An error occurred during the prediction process."

if __name__ == '__main__':
    app.run(debug=True)
