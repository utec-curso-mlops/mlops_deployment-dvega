import os

DEFAULT_BUCKET = "mlops-utec"
ENV_CODE = "prod"
TRACKING_SERVER_ARN = 'arn:aws:sagemaker:us-east-1:654654589924:mlflow-tracking-server/mlops-utec-mlflow-server'
USERNAME = os.getenv("GITHUB_ACTOR")



