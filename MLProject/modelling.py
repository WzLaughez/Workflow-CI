import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import random
import numpy as np
 
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
 
# Create a new MLflow Experiment
mlflow.set_experiment("Latihan Churn Prediction")
mlflow.autolog()
 
data = pd.read_csv("processed.csv")
 
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

input_example = X_train.iloc[:5] 

with mlflow.start_run():
    # Log parameters
    # Train model
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )
    model.fit(X_train, y_train)
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )