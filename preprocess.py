import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='matches.csv')
    parser.add_argument('--outputpath_train', type=str, default='/opt/ml/processing/output/train/')
    parser.add_argument('--outputpath_test', type=str, default='/opt/ml/processing/output/test/')
    parser.add_argument('--outputpath_val', type=str, default='/opt/ml/processing/output/validation/')
    
    return parser.parse_known_args()

if __name__ == "__main__":
    args, _ = _parse_args()
    logger.info("Arguments parsed. Filepath: %s, Filename: %s, Outputpath Train: %s, Outputpath Test: %s, Outputpath Val: %s", 
                args.filepath, args.filename, args.outputpath_train, args.outputpath_test, args.outputpath_val)

    # Load the data
    df = pd.read_csv(os.path.join(args.filepath, args.filename))
    logger.info("Data loaded from %s", os.path.join(args.filepath, args.filename))
    
    # Preprocessing
    df.dropna(inplace=True)  # Drop rows with missing values
    df.drop(["date", "time", "day", "referee", "match report", "captain"], axis=1, inplace=True, errors='ignore')
    df['round'] = df['round'].str.replace('Matchweek ', '').astype(int)
    df['comp'] = 1
    df['venue'] = df['venue'].replace({'Home': 1, 'Away': 2})
    df['result'] = df['result'].replace({'W': 3, 'D': 1, 'L': 0})
    df = pd.get_dummies(df, columns=['opponent', 'formation', 'team'], drop_first=True)
    logger.info("Preprocessing completed. Data columns: %s", df.columns)

    # Train-Test Split
    # First, split the data into 70% train and 30% temp
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=7, stratify=df['result'])

    # Then, split the temp data into 50% validation and 50% test (which is 15% each of the original data)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=7, stratify=temp_data['result'])

    logger.info("Train-Validation-Test split completed. Train size: %d, Validation size: %d, Test size: %d", 
                len(train_data), len(val_data), len(test_data))

    # Save to local CSV files
    os.makedirs(args.outputpath_train, exist_ok=True)
    os.makedirs(args.outputpath_val, exist_ok=True)
    os.makedirs(args.outputpath_test, exist_ok=True)

    train_file = os.path.join(args.outputpath_train, 'matches_train.csv')
    val_file = os.path.join(args.outputpath_val, 'matches_val.csv')
    test_file = os.path.join(args.outputpath_test, 'matches_test.csv')

    # Save train, validation, and test data
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    test_data.to_csv(test_file, index=False)

    logger.info("Train, validation, and test data saved to %s, %s, and %s", train_file, val_file, test_file)

    print("## Preprocessing and uploading completed. Exiting.")
