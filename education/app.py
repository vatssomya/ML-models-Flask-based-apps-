from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model (make sure it's in the correct path)
model = pickle.load(open('best_model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        SCHTOT = float(request.form['SCHTOT'])
        SCH1 = float(request.form['SCH1'])
        SCH2 = float(request.form['SCH2'])
        SCH3 = float(request.form['SCH3'])
        TOTPOPULAT = float(request.form['TOTPOPULAT'])
        P_SC_POP = float(request.form['P_SC_POP'])
        P_ST_POP = float(request.form['P_ST_POP'])
        SEXRATIO = float(request.form['SEXRATIO'])
        OVERALL_LI = float(request.form['OVERALL_LI'])
        FEMALE_LIT = float(request.form['FEMALE_LIT'])
        MALE_LIT = float(request.form['MALE_LIT'])
        TCHTOT = float(request.form['TCHTOT'])
        TCHTOTG = float(request.form['TCHTOTG'])
        TCHTOTM = float(request.form['TCHTOTM'])
        AREA_SQKM = float(request.form['AREA_SQKM'])

        # Prepare the input data in the correct shape for the model
        input_data = np.array([[SCHTOT, SCH1, SCH2, SCH3, TOTPOPULAT, P_SC_POP, P_ST_POP, SEXRATIO, 
                                OVERALL_LI, FEMALE_LIT, MALE_LIT, TCHTOT, TCHTOTG, TCHTOTM, AREA_SQKM]])

        # Make a prediction
        prediction = model.predict(input_data)

        # Send the prediction to the result page
        prediction_text = f"The predicted enrollment total is: {prediction[0]:,.2f}"

        return render_template('result.html', prediction_text=prediction_text)

    except Exception as e:
        # Handle errors (e.g., missing or invalid input)
        return render_template('index.html', error_message="Error: Please check your inputs and try again.")

if __name__ == "__main__":
    app.run(debug=True)
