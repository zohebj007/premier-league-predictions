# Import Necessary Libraries
from __future__ import print_function
import argparse
import os
import pandas as pd
import json
import numpy as np
from sklearn import ensemble
import logging
import pickle
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Training Model
def model(args, x_train, y_train):
    # Initialize the model with parameters
    model = ensemble.RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
    # Train the model
    model.fit(x_train, y_train)
    
    logger.info("Training Accuracy: {:.3f}".format(model.score(x_train, y_train)))
    
    return model

# Load Training Data
def load_train_data(file_path):
    # Read the training data from the CSV file
    df = pd.read_csv(os.path.join(file_path, "matches_train.csv"))
    
    # Extract features (all columns except 'result')
    features = df.drop(columns=['result'])
    
    # Extract labels (the 'result' column)
    label = df['result']
    
    return features, label

# Parse Command-Line Arguments
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unknown = _parse_args()
    # Load training data
    train_data, train_labels = load_train_data(args.train)
    # Train the classifier model
    classifier = model(args, train_data, train_labels)
    
    # Save the trained model to the specified directory
    with open(os.path.join(args.model_dir, "model.pkl"), 'wb') as model_file:
        pickle.dump(classifier, model_file)

# Define Model Deserialization Function
def model_fn(model_dir):
    with open(os.path.join(model_dir, "model.pkl"), 'rb') as model_file:
        model = pickle.load(model_file)
    return model
