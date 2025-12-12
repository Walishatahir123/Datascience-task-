from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib

app = Flask(__name__)

model = joblib.load(r"C:\Users\user\Desktop\datascience\.dist\flaskm\random_forest_model.pkl")

# # Load trained Random Forest model
# model = joblib.load("random_forest_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        lotarea = float(request.form["lotarea"])
        overallquality = float(request.form["overallquality"])
        yearbuilt = float(request.form["yearbuilt"])
        TotalBsmtSF = float(request.form["TotalBsmtSF"])
        GrLivArea = float(request.form["GrLivArea"])
        GarageCars = float(request.form["GarageCars"])

        # Arrange in correct order for the model
        features = np.array([[lotarea, overallquality, yearbuilt, TotalBsmtSF, GrLivArea, GarageCars]])

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template("index.html", prediction_text=f"Predicted Price Category: {prediction}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
