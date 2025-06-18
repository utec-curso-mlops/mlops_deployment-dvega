import sagemaker
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir)))
from utils import DEFAULT_BUCKET, ENV_CODE, TRACKING_SERVER_ARN, USERNAME, ENV_CODE

# Sagemaker configuration
SAGEMAKER_ROLE = "arn:aws:iam::654654589924:role/service-role/SageMaker-MLOpsEngineer"
default_prefix = f"sagemaker/credit-card-fraud-detection/{USERNAME}"
DEFAULT_PATH = DEFAULT_BUCKET + "/" + default_prefix
sagemaker_session = sagemaker.Session(default_bucket=DEFAULT_BUCKET,
                                      default_bucket_prefix=default_prefix)
#Pipeline configuration
PIPELINE_NAME = f"pipeline-train-{ENV_CODE}-{USERNAME}"
MODEL_NAME = f"credit-card-fraud-detection-{USERNAME}"
TRACKING_SERVER_ARN = TRACKING_SERVER_ARN
USERNAME = USERNAME
ENV_CODE = ENV_CODE