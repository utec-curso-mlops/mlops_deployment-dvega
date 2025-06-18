import boto3
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir)))
from faas_utils import FUNCTION_NAME

def test_lambda_invocation():
    client = boto3.client('lambda')
    payload = {"body": {"data": [0,237.10,487.97,504.7,7878.1]}}
    response = client.invoke(FunctionName=FUNCTION_NAME,Payload=json.dumps(payload),LogType='Tail')
    response_payload = json.loads(response['Payload'].read().decode('utf-8'))
    assert abs(eval(response_payload["body"])["prediction"] - 0.0007983) < 0.000001

