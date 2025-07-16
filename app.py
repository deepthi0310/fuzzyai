from flask import Flask, render_template, request
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import joblib
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load ML model
ml_model = joblib.load("model.pkl")

# Define fuzzy logic system
ph = ctrl.Antecedent(np.arange(0, 14, 0.1), 'ph')
turbidity = ctrl.Antecedent(np.arange(0, 100, 1), 'turbidity')
temperature = ctrl.Antecedent(np.arange(0, 50, 1), 'temperature')
dosage = ctrl.Consequent(np.arange(0, 100, 1), 'dosage')

ph.automf(3)
turbidity.automf(3)
temperature.automf(3)

dosage['low'] = fuzz.trimf(dosage.universe, [0, 0, 50])
dosage['medium'] = fuzz.trimf(dosage.universe, [25, 50, 75])
dosage['high'] = fuzz.trimf(dosage.universe, [50, 100, 100])

rule1 = ctrl.Rule(ph['poor'] | turbidity['poor'] | temperature['poor'], dosage['high'])
rule2 = ctrl.Rule(ph['average'] | turbidity['average'] | temperature['average'], dosage['medium'])
rule3 = ctrl.Rule(ph['good'] | turbidity['good'] | temperature['good'], dosage['low'])

dosing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
dosing = ctrl.ControlSystemSimulation(dosing_ctrl)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ph_val = float(request.form['ph'])
    turb_val = float(request.form['turbidity'])
    temp_val = float(request.form['temperature'])
    mode = request.form.get('mode')

    if mode == 'ml':
        result = ml_model.predict([[ph_val, turb_val, temp_val]])[0]
    else:
        dosing.input['ph'] = ph_val
        dosing.input['turbidity'] = turb_val
        dosing.input['temperature'] = temp_val
        dosing.compute()
        result = dosing.output['dosage']

    # Save the prediction
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ph": ph_val,
        "turbidity": turb_val,
        "temperature": temp_val,
        "dosage": round(result, 2),
        "mode": mode
    }

    df = pd.DataFrame([record])
    if os.path.exists("prediction_history.csv"):
        df.to_csv("prediction_history.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("prediction_history.csv", index=False)

    return render_template('result.html', dosage=round(result, 2))

@app.route('/history')
def history():
    if os.path.exists("prediction_history.csv"):
        df = pd.read_csv("prediction_history.csv")
        return render_template('history.html', tables=[df.to_html(classes='table table-bordered', index=False)], titles=df.columns.values)
    else:
        return "<h3>No history available yet.</h3>"

@app.route('/graph')
def graph():
    if not os.path.exists("prediction_history.csv"):
        return "<h3>No data to graph.</h3>"

    df = pd.read_csv("prediction_history.csv")

    plt.figure(figsize=(6,4))
    plt.scatter(df["ph"], df["dosage"], c='blue', label='Dosage by pH')
    plt.xlabel("pH")
    plt.ylabel("Dosage")
    plt.title("pH vs Dosage")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f'<img src="data:image/png;base64,{graph_url}"/>'

if __name__ == "__main__":
    app.run(debug=True)
