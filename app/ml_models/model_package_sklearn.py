import numpy as np
import os
import sklearn
import joblib


class model():
    def __init__(self, ModelPath, ModelName):
        self.pipeline = joblib.load(
            open(os.path.join(os.getcwd(), ModelPath, ModelName), "rb"))

    def get_top_k(self, text, k):
        text = [text]

        predictions = self.pipeline.predict_proba(text)

        best_k = np.argsort(predictions, axis=1)[:, -k:]

        # dictionnary of predicted classes with their probabilities
        results = {
            self.pipeline.steps[1][1].classes_[i]: "{:12.2f}%".format(
                float(predictions[0][i]) * 100)
            for i in best_k[0][::-1]
        }
        return results


if __name__ == "__main__":
    name = input("Enter model name (with extension):")
    text = input("Enter text to be predicted:")
    Model = model(r'model', name)
    print(Model.get_top_k(text, 3))
