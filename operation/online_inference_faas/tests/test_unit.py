import boto3
import pytest
from botocore.exceptions import ClientError

def s3_path_exists(bucket_name, key):
    """Checks if an S3 object (path) exists."""
    s3_client = boto3.client("s3")
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

@pytest.mark.parametrize("bucket_name, key, expected", [
    ("mlops-utec", "mlflow-server/1/52ce21cdfba8475dae1bfafd337609e4/artifacts/model/model.xgb", True),  # Object exists
    ("mlops-utec", "mlflow-server/1/52ce21cdfba8475dae1bfafd337609e4/artifacts/model/model.torch", False) # Object does not exist
])
def test_s3_path(bucket_name, key, expected):
    """Test if an S3 path exists or not."""
    assert s3_path_exists(bucket_name, key) == expected, f"Path '{key}' existence does not match expected value!"

