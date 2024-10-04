pip install dvc[s3]
#which dvc
#export PATH=!$PATH:/opt/conda/bin/dvc

######################  imports  #################################
from sagemaker import Session
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline_context import PipelineSession
import boto3
import pandas as pd
import numpy as np
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sagemaker.sklearn import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.steps import ProcessingStep, CreateModelStep
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.model import Model
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.model_step import ModelStep

###git####
#!git fetch --tags
#latest_tag = !git describe --tags `git rev-list --tags --max-count=1`
# Check what the latest tag is
#print(f"Latest tag: {latest_tag[0]}")
# Checkout the latest tag
#!git checkout {latest_tag[0]}
#!dvc pull
#df = pd.read_csv('matches.csv')
#df.head()

#git fetch
git checkout dvc-branch
dvc pull
#df = pd.read_csv('matches.csv')
#df.columns

#####s3 urls######
bucket='zohebmlops'
prefix = 'mlops'
###preprocessing paths #####
input_source = "matches.csv"
train_path = f"s3://{bucket}/{prefix}/train"
test_path = f"s3://{bucket}/{prefix}/test"
val_path = f"s3://{bucket}/{prefix}/val"
####training paths###
model_output_uri = "s3://{}/{}/output".format(bucket, prefix)
train_data_uri = train_path
test_data_uri = test_path
train_input = TrainingInput(s3_data=train_data_uri, content_type='text/csv')
#####eval paths
evaluation_output_uri = f"s3://{bucket}/output/evaluation"


################# important for pipeline #####################
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
pipeline_session = PipelineSession()


#######preprocessing########
sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1, 
    base_job_name='macthes-main'
)

# Define processing step
processing_step = ProcessingStep(
    name='PreprocessingStep',
    processor=sklearn_processor,
    code='preprocess.py',  # Path to your preprocessing script
    inputs=[
        ProcessingInput(
            source=input_source, 
            destination="/opt/ml/processing/input",
            s3_input_mode="File",
            s3_data_distribution_type="ShardedByS3Key",
            
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train_data", 
            source="/opt/ml/processing/output/train",
            destination=train_path,
            s3_upload_mode="EndOfJob",
        ),
        ProcessingOutput(
            output_name="test_data", 
            source="/opt/ml/processing/output/test",
            destination=test_path,
            s3_upload_mode="EndOfJob",
        ),
        ProcessingOutput(
            output_name="val_data", 
            source="/opt/ml/processing/output/validation",
            destination=val_path,
            s3_upload_mode="EndOfJob",
        ),
    ]
)

######training########
########  train step ###########

estimator = SKLearn(entry_point='train.py',
                    framework_version="0.23-1",
                    py_version='py3',
                    instance_type='ml.m5.xlarge',
                    role=role,
                    output_path=model_output_uri,
                    base_job_name='sklearn-matches',
                    hyperparameters={'n_estimators': 50, 'max_depth': 5})
# Define the input data for training and testing
train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "training": train_input,
    },
)

#######eval#######

###evall
##### train artifacts ######
model_artifact_path = train_step.properties.ModelArtifacts.S3ModelArtifacts

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)


eval_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1, 
    base_job_name='eval-match-main'
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=eval_processor,
    inputs=[
        ProcessingInput(source=model_artifact_path, destination="/opt/ml/processing/model"),
        ProcessingInput(source=test_path, destination="/opt/ml/processing/test"),
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/output", destination=evaluation_output_uri),
    ],
    code="eval.py",
    property_files=[evaluation_report]
)

# Define ModelMetrics for registration (optional, to track accuracy and other metrics)
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=f"{evaluation_output_uri}/evaluation.json",
        content_type="application/json"
    )
)

#######model-creation######

sklearn_model = SKLearnModel(
    model_data=model_artifact_path,  # S3 path to your model.tar.gz
    role=role,  # Your IAM role ARN
    framework_version='0.23-1',  # The scikit-learn version that matches your training
    sagemaker_session=pipeline_session  # Your current SageMaker session
)

register_args = sklearn_model.register(
    content_types=["application/x-model"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="priemier-league",  # Specify the model group name here
    model_metrics=model_metrics  # Optional: Attach model metrics
)
step_register = ModelStep(name="premier-league", step_args=register_args)

#####condition check##########

# Define the condition to check accuracy
cond_gte = ConditionGreaterThan(
    left=JsonGet(
        step_name=evaluation_step.name,
        property_file=evaluation_report,
        json_path="accuracy"
    ),
    right=0.65
)

# Create a condition step
condition_step = ConditionStep(
    name="CheckAccuracy",
    conditions=[cond_gte],
    if_steps=[step_register],  # Register the model if accuracy > 60%
    else_steps=[]  # Do nothing if the accuracy is <= 60%
)

train_step.add_depends_on([processing_step])
evaluation_step.add_depends_on([train_step])

# Define the pipeline
pipeline = Pipeline(
    name="preimier-league-prediction-pipeline",
    steps=[processing_step, train_step, evaluation_step, condition_step],
    sagemaker_session=pipeline_session,  # Ensure the session is passed here
)

# Create and start the pipeline using SageMaker client
#pipeline.create(role_arn=role)  # Replace with your actual role ARN
pipeline.start()
