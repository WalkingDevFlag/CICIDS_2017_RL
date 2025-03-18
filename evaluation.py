# evaluation.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_agent(env, agent):
    """
    Evaluate the trained agent on the environment.
    
    Returns a dictionary with evaluation metrics.
    """
    states = []
    predictions = []
    true_labels = []
    
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        true_label = env.y[env.current_index] if env.current_index < len(env.y) else None
        predictions.append(action)
        true_labels.append(true_label)
        states.append(state)
        state, _, done, _ = env.step(action)
    
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    rec = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(true_labels, predictions)
    
    try:
        roc_auc = roc_auc_score(true_labels, predictions, average='weighted', multi_class='ovo')
    except Exception:
        roc_auc = None
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": conf_mat,
        "roc_auc": roc_auc
    }
    
    logger.info("Evaluation Metrics: %s", metrics)
    return metrics
