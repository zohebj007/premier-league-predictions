import argparse
import os
import pickle
import tarfile
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score, classification_report
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--test', type=str, default='/opt/ml/processing/test')
    return parser.parse_args()

def load_model(model_dir):
    model_tar_path = os.path.join(model_dir, 'model.tar.gz')
    
    if not os.path.exists(model_tar_path):
        raise FileNotFoundError(f"Model tar.gz file not found at {model_tar_path}")
    
    with tarfile.open(model_tar_path, 'r:gz') as tar:
        tar.extractall(path=model_dir)
    
    model_path = os.path.join(model_dir, 'model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    return model

def load_test_data(file_path):
    df = pd.read_csv(os.path.join(file_path, "matches_test.csv"))
    features = df.drop(columns=['result'])
    labels = df['result']
    return features, labels

if __name__ == "__main__":
    args = _parse_args()

    logger.info("Loading the trained model")
    model = load_model(args.model_dir)

    logger.info("Loading test data")
    test_features, test_labels = load_test_data(args.test)

    logger.info("Making predictions")
    predictions = model.predict(test_features)

    logger.info("Evaluating the model")
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, output_dict=True)

    evaluation_output_dir = args.output_data_dir
    os.makedirs(evaluation_output_dir, exist_ok=True)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")

    logger.info("Writing evaluation results")
    with open(os.path.join(evaluation_output_dir, "evaluation.json"), "w") as f:
        json.dump({"accuracy": accuracy, "classification_report": report}, f, indent=2)

    with open(os.path.join(evaluation_output_dir, "accuracy.json"), "w") as f:
        json.dump({"accuracy": float(accuracy)}, f)  # Ensure accuracy is JSON serializable

    logger.info("Evaluation complete")
