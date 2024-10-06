# Create a virtual environment in the Jenkins workspace
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install the required dependencies inside the virtual environment
pip install sagemaker boto3 scikit-learn

# Run the SageMaker pipeline script
python pipeline.py

# Deactivate the virtual environment after the job is done
deactivate
