# data_loader.py
import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(csv_path, stack_size=4):
    """
    Load the preprocessed CICIDS2017 CSV file, normalize features,
    encode labels, and stack temporal features.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        stack_size (int): Number of consecutive timesteps to stack.
        
    Returns:
        X (np.array): The state observations stacked (num_samples, stack_size * num_features).
        y (np.array): The encoded labels.
    """
    if not os.path.exists(csv_path):
        logger.error("CSV file not found at: %s", csv_path)
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("Loading data from %s", csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.exception("Error reading CSV file: %s", e)
        raise

    # Ensure columns match expected format
    feature_columns = [col for col in df.columns if col.startswith("PC")]
    label_column = "Attack Type"
    
    if label_column not in df.columns or not feature_columns:
        logger.error("Dataset does not contain expected columns.")
        raise ValueError("Missing required columns in dataset.")

    # Extract features and labels
    features = df[feature_columns].values
    labels = df[label_column].values

    # Normalize features
    logger.info("Normalizing features...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Encode labels using LabelEncoder
    logger.info("Encoding labels...")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Temporal feature stacking
    logger.info("Performing temporal feature stacking with stack_size=%d ...", stack_size)
    num_samples, num_features = features.shape
    if num_samples < stack_size:
        logger.error("Not enough samples (%d) to stack %d timesteps.", num_samples, stack_size)
        raise ValueError("Insufficient samples for stacking.")

    stacked_features = []
    stacked_labels = []
    
    # Use tqdm for progress tracking during stacking
    for i in tqdm(range(num_samples - stack_size + 1), desc="Stacking features"):
        stacked_state = features[i:i+stack_size].flatten()
        stacked_features.append(stacked_state)
        stacked_labels.append(labels[i + stack_size - 1])
    
    X = np.array(stacked_features)
    y = np.array(stacked_labels)

    logger.info("Data loaded: stacked X shape: %s, y shape: %s", X.shape, y.shape)
    
    return X, y, label_encoder.classes_

if __name__ == "__main__":
    # For testing purposes
    X, y, class_names = load_and_preprocess_data("cicids2017_preprocessed.csv")
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Class names:", class_names)
