# Use an AWS Lambda-compatible base image
FROM public.ecr.aws/lambda/python:3.8
#FROM public.ecr.aws/sagemaker/sagemaker-distribution:3.1-cpu

# Install dependencies
RUN pip install --no-cache-dir mlflow==2.13.2 sagemaker-mlflow==0.1.0 xgboost==2.1.4 boto3 numpy pandas 

# Copy entrypoint script
COPY entrypoint.py /var/task/entrypoint.py

# Set entrypoint
CMD ["entrypoint.lambda_handler"]