import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model
model = load_model("model/ann_model.h5")

# Prepare scaler
data = pd.read_csv("dataset/student_data.csv")
X = data.drop("result", axis=1)

scaler = StandardScaler()
scaler.fit(X)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""

    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_marks = float(request.form["previous_marks"])
        assignments = int(request.form["assignments"])

        student_data = np.array([[study_hours, attendance, previous_marks, assignments]])
        student_data = scaler.transform(student_data)

        prediction = model.predict(student_data)

        if prediction > 0.5:
            prediction_text = "🎓 Student is likely to PASS"
        else:
            prediction_text = "❌ Student is likely to FAIL"

    return render_template("index.html", prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
