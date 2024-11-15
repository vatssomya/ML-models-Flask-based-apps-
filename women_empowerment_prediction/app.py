from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load("best_model.sav")

# Feature columns expected by the model
feature_columns = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 
                   'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather input data from form
        input_features = [float(request.form.get(feature, 0)) for feature in feature_columns]  # default to 0 if missing
        # Predict using the loaded model
        prediction = model.predict([input_features])[0]
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        print(f"Error: {e}")
        return redirect(url_for('index'))  # Redirect to home on error

if __name__ == '__main__':
    app.run(debug=True)
