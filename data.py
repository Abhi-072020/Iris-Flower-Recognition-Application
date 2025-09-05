# data.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=0.2, random_state=42)
