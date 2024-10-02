import boto3
import pandas as pd
from sagemaker.transformer import Transformer

# Set up the SageMaker client
sm_client = boto3.client("sagemaker")

# Get the latest approved model package from the model package group
model_package_group_name = "priemier-league"  # Update with your actual model package group name
response = sm_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1,
    ModelApprovalStatus="Approved"
)

# Extract the model package ARN
model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
print(f"Latest approved model ARN: {model_package_arn}")

# Load your batch data
batch_data = pd.read_csv("batch-data.csv")
preprocessed_data_s3_key = 'batch-data/batch-data.csv'

# Upload batch data to S3
s3_client = boto3.client("s3")
s3_client.upload_file("batch-data.csv", bucket, preprocessed_data_s3_key)
data_path_batch = f"s3://{bucket}/{preprocessed_data_s3_key}"
print(f"Batch data S3 path: {data_path_batch}")

# Define the transformer using the model package ARN
transformer = Transformer(
    model_name=model_package_arn,  # Use the full model package ARN
    instance_count=1,
    instance_type="ml.m5.large",
    strategy="MultiRecord",  # Optional: MultiRecord improves throughput for large datasets
    output_path=f"s3://{bucket}/{prefix}/batch_output",  # Specify your S3 output path
    assemble_with="Line",  # How the output will be assembled
    accept="application/jsonlines"  # Define the content type for the output
)

# Run batch transform job on your dataset
transformer.transform(
    data=data_path_batch,  # S3 path to your validation or batch inference dataset
    content_type="text/csv",  # Content type of input data
    split_type="Line",  # Split type of input data
    input_filter="$[1:]",  # Optional: Ignore the 'result' column or specific columns
    join_source="Input",  # Optional: Adds input features to the output
    output_filter="$[0,-1]"  # Optional: Filters out unwanted columns in the output
)

# Start the batch transform job and wait for it to complete
transformer.wait()
