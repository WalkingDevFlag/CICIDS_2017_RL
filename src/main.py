# main.py
import os
import logging
import numpy as np
import torch
import gym

from data_loader import load_and_preprocess_data
from custom_env import CIDS2017Env
from agent import DQNAgent
from training import train_agent
from evaluation import evaluate_agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load the data (adjust stack_size if needed)
        X, y, class_names = load_and_preprocess_data("cicids2017_preprocessed.csv", stack_size=4)
        logger.info("Class names: %s", class_names)
    except Exception as e:
        logger.exception("Failed to load data: %s", e)
        return

    # Create the Gym environment with dynamic reward scaling and cost-sensitive penalty
    env = CIDS2017Env(X, y, dynamic_reward_scale=1.0, cost_penalty=2.0)

    # Determine input dimension and number of actions (classes)
    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Optional: Define cost weights per class (here kept uniform)
    cost_weights = np.ones(num_actions)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Initialize the DQN agent
    agent = DQNAgent(input_dim, num_actions, lr=1e-3, gamma=0.99,
                     epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                     cost_weights=cost_weights, device=device)

    # Train the agent
    trained_agent = train_agent(env, agent, num_episodes=50, batch_size=64, replay_capacity=5000, target_update=5)
    
    # Reinitialize environment for evaluation
    env.current_index = 0

    # Evaluate the trained agent
    metrics = evaluate_agent(env, trained_agent)
    logger.info("Final Evaluation Metrics: %s", metrics)

if __name__ == "__main__":
    main()
