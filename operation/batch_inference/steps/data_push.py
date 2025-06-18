from batch_inference_utils import TRACKING_SERVER_ARN, DEFAULT_PATH, SAGEMAKER_ROLE, MODEL_NAME, USERNAME
from sagemaker.workflow.function_step import step

# Global variables
instance_type = "ml.m5.large"
image_uri = "885854791233.dkr.ecr.us-east-1.amazonaws.com/sagemaker-distribution-prod@sha256:92cfd41f9293e3cfbf58f3bf728348fbb298bca0eeea44464968f08622d78ed0"

@step(
    name="DataPush",
    instance_type=instance_type,
    image_uri=image_uri,
    role=SAGEMAKER_ROLE
)
def data_push(inf_proc_s3_path: str,experiment_name: str,run_id: str, cod_month: int):
    import subprocess
    subprocess.run(['pip', 'install', 'awswrangler==3.12.0']) 
    import pandas as pd
    import mlflow
    import numpy as np
    from datetime import datetime
    import pytz
    import awswrangler as wr

    ID_COL = "transaction_id"
    TIME_COL = "cod_month"
    PRED_COL = "prob"
    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(inf_proc_s3_path)
    df['fraud_profile'] = np.where(df[PRED_COL] >= 0.415, 'High risk',
                                   np.where(df[PRED_COL] >= 0.285, 'Medium risk',
                                   'Low risk'))

    df['model'] = MODEL_NAME
    timezone = pytz.timezone("America/Lima")
    df['load_date'] = datetime.now(timezone).strftime("%Y%m%d")
    df['order'] = df.prob.rank(method='first', ascending=False).astype(int)

    inf_posproc_s3_path = f"s3://{DEFAULT_PATH}/inf-posproc-data"
    inf_posproc_s3_path_partition = inf_posproc_s3_path + f'/{TIME_COL}={cod_month}/output.parquet'
    database = 'risk_management'
    table_name = database + f'.fraud_detection_{USERNAME}'

    # Pushing data to S3 path
    df = df[[ID_COL, PRED_COL, 'model','fraud_profile','load_date', 'order', TIME_COL]] 
    df.to_parquet(inf_posproc_s3_path_partition, engine='pyarrow', compression='snappy')

    # Creating table
    ddl = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {table_name} (
    {ID_COL} string,
    {PRED_COL} double,
    model string,
    fraud_profile string,
    load_date string,
    order int
    )
    PARTITIONED BY ({TIME_COL} int)
    STORED AS parquet
    LOCATION '{inf_posproc_s3_path}'
    TBLPROPERTIES ('parquet.compression'='SNAPPY')
    """
    query_exec_id = wr.athena.start_query_execution(sql=ddl, database=database)
    wr.athena.wait_query(query_execution_id=query_exec_id)

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="DataPush", nested=True):
                mlflow.log_input(
                mlflow.data.from_pandas(df, inf_posproc_s3_path_partition),
                context="DataPush"
            )
    # Refreshing partition
    dml = f"MSCK REPAIR TABLE {table_name}"
    query_exec_id = wr.athena.start_query_execution(sql=dml, database=database)
    wr.athena.wait_query(query_execution_id=query_exec_id)