import os
import mlflow

# MLflow ko file system use karne ki ijazat dene ke liye yeh line add karen
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

mlflow.set_experiment("Plant_Disease_Detection")
with mlflow.start_run():
    mlflow.log_param("test_param", "worked")
    print("Test run logged successfully!")