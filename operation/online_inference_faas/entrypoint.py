import mlflow
import json
#from faas_utils import TRACKING_SERVER_ARN, MODEL_NAME, MODEL_VERSION

# Load model
# Load model
model_name = "credit-card-fraud-detection"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"

# Set tracking server
tracking_server_arn = 'arn:aws:sagemaker:us-east-1:654654589924:mlflow-tracking-server/mlops-utec-mlflow-server'

#model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

# Set tracking server
#mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
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