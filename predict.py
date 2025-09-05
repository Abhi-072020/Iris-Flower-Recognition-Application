# predict.py

import joblib
import pandas as pd

# Label mapping from number to name
label_map = {
    0: "Iris setosa",
    1: "Iris versicolor",
    2: "Iris virginica"
}

def predict(sample):
    model = joblib.load("models/model.pkl")
    df = pd.DataFrame([sample], columns=["sepal length (cm)", "sepal width (cm)",
                                         "petal length (cm)", "petal width (cm)"])
    prediction = model.predict(df)
    class_name = label_map[prediction[0]]

    
    print(f"Predicted class: {prediction[0]} ({class_name})")

if __name__ == "__main__":
    sample_input = [6.0, 2.2, 4.0, 1.0]  # Example input
    predict(sample_input)
