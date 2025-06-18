import mlflow
import json

# Load model
model_name = "credit-card-fraud-detection"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"

# Set tracking server
tracking_server_arn = 'arn:aws:sagemaker:us-east-1:654654589924:mlflow-tracking-server/mlops-utec-mlflow-server'
mlflow.set_tracking_uri(tracking_server_arn)
model = mlflow.xgboost.load_model(model_uri)


def lambda_handler(event, context):
    if not isinstance(event, dict):
        event = eval(event)
    data = [event['body']['data']]
    pred = model.predict_proba(data)[:, 1][0]

    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': float(pred)})
    }