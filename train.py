# train.py

from data import load_data
from model import train_model

def main():
    X_train, X_test, y_train, y_test = load_data()
    train_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
