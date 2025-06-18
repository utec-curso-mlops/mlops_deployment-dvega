from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import ParameterInteger
from steps.data_pull import data_pull
from steps.model_inference import model_inference
from steps.data_push import data_push


from batch_inference_utils import MODEL_NAME, USERNAME, ENV_CODE, PIPELINE_NAME, SAGEMAKER_ROLE

#MLFlow setting
experiment_name = f"pipeline-inference-{ENV_CODE}-{USERNAME}"

# Parameter setting
cod_month = ParameterInteger(name="PeriodoCarga")

# Steps setting
data_pull_step = data_pull(experiment_name=experiment_name,
                           run_name=ExecutionVariables.PIPELINE_EXECUTION_ID,
                           cod_month=cod_month)

model_inference_step = model_inference(inf_raw_s3_path=data_pull_step[0],
                                     experiment_name=data_pull_step[1],
                                     run_id=data_pull_step[2],
                                       cod_month=cod_month)

data_push_step = data_push(inf_proc_s3_path=model_inference_step[0],
                                     experiment_name=model_inference_step[1],
                                     run_id=model_inference_step[2],
                                      cod_month=cod_month)


# Pipeline creation
pipeline = Pipeline(name=PIPELINE_NAME,
                    steps=[data_pull_step, model_inference_step,data_push_step],
                    parameters=[cod_month])
pipeline.upsert(role_arn=SAGEMAKER_ROLE)

