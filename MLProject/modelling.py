import os
import sys
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)


    # Ambil path dari argumen atau default ke processed.csv
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed.csv")
    data = pd.read_csv(file_path)

    # Pisahkan fitur dan target
    X = data.drop("target", axis=1)
    y = data["target"]

    # Bagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    # Contoh input untuk keperluan MLflow
    input_example = X_train.iloc[:5]

    # Ambil parameter dari CLI atau default
    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    random_state = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    # Mulai tracking run
    with mlflow.start_run(nested=True):
        model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # Log metrik
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
