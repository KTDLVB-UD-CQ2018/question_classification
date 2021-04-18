from ml_models.model_package_sklearn import model
from flask import Flask, request, jsonify, render_template

import numpy as np
import pandas as pd

import random
import time
import os

# initialize flask application
app = Flask(__name__)

# Fixing all seeds


def random_seed(seed_value):
    np.random.seed(seed_value)  # cpu vars
    random.seed(seed_value)  # Python


random_seed(500)


model_nb = model(r'ml_models', "pipeline_nb.joblib")
model_svm = model(r'ml_models', "pipeline_svm.joblib")
model_logreg = model(r'ml_models', "pipeline_logreg.joblib")

# Define API
HOST = "0.0.0.0"
PORT = os.environ.get("PORT", 8080)

data = pd.read_csv("dataset.csv", encoding='utf-8')
df_test = data.loc[data['DATASET'] == 'TEST']


@app.route("/")
def index():
    return render_template("index.html", prediction={})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
        For rendering results on HTML GUI
        """
    if request.method == "POST":

        if request.form["submit_button"] == "Generate":
            i = random.sample(list(df_test.index), 1)[0]
            question_text = df_test["Câu hỏi"][i]
            return render_template("index.html", prediction={}, news=str(question_text))

        elif request.form["submit_button"] == "Multinomial Naive Bayes":
            text_in = request.form["text"]
            if len(text_in) == 0:
                return render_template(
                    "index.html",
                    prediction={"ERROR": "Không có nội dung"},
                )
            else:
                # Check if text_in is in inference data
                filter1 = data["Câu hỏi"].isin([text_in])
                true_value = (
                    data["LOẠI"][filter1].values[0]
                    if filter1.any()
                    else ": Không tồn tại trong bộ dữ liệu"
                )
                output = model_nb.get_top_k(text_in, 3)
                return render_template(
                    "index.html",
                    prediction=output,
                    news=str(text_in),
                    real=str(true_value),
                )

        elif request.form["submit_button"] == "Support Vector Machine":
            text_in = request.form["text"]
            if len(text_in) == 0:
                return render_template(
                    "index.html",
                    prediction={"ERROR": "Không có nội dung"},
                )
            else:
                # Check if text_in is in inference data
                filter1 = data["Câu hỏi"].isin([text_in])
                true_value = (
                    data["LOẠI"][filter1].values[0]
                    if filter1.any()
                    else ": Không tồn tại trong bộ dữ liệu"
                )
                output = model_svm.get_top_k(text_in, 3)
                return render_template(
                    "index.html",
                    prediction=output,
                    news=str(text_in),
                    real=str(true_value),
                )

        elif request.form["submit_button"] == "Linear Regression":
            text_in = request.form["text"]
            if len(text_in) == 0:
                return render_template(
                    "index.html",
                    prediction={"ERROR": "Không có nội dung"},
                )
            else:
                filter1 = data["Câu hỏi"].isin([text_in])
                true_value = (
                    data["LOẠI"][filter1].values[0]
                    if filter1.any()
                    else ": Không tồn tại trong bộ dữ liệu"
                )
                output = model_logreg.get_top_k(text_in, 3)
                return render_template(
                    "index.html",
                    prediction=output,
                    news=str(text_in),
                    real=str(true_value),
                )

        elif request.form["submit_button"] == "Clear":
            return render_template("index.html", prediction={}, news="")

        else:
            pass

    elif request.method == "GET":
        return render_template("index.html", prediction={})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)
