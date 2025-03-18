# training.py
import numpy as np
import torch
import logging
from agent import DQNAgent
from memory import PrioritizedReplayBuffer
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_agent(env, agent, num_episodes=100, batch_size=64, replay_capacity=10000, target_update=10):
    """
    Train the DQN agent in the given environment.
    
    Parameters:
        env (gym.Env): The RL environment.
        agent (DQNAgent): The DQN agent.
        num_episodes (int): Number of training episodes.
        batch_size (int): Minibatch size.
        replay_capacity (int): Capacity of the replay buffer.
        target_update (int): Frequency (episodes) for target network update.
        
    Returns:
        agent (DQNAgent): The trained agent.
    """
    replay_buffer = PrioritizedReplayBuffer(replay_capacity)
    beta_start = 0.4
    beta_frames = num_episodes * 100  # approximate beta annealing over training frames
    total_steps = 0

    # Wrap episodes with tqdm progress bar.
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state = env.reset()
        episode_reward = 0
        done = False
        # Create a progress bar for steps in the current episode.
        step_bar = tqdm(total=env.num_samples, leave=False, desc=f"Episode {episode+1} Steps")
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1
            step_bar.update(1)
            
            beta = min(1.0, beta_start + total_steps * (1.0 - beta_start) / beta_frames)
            if len(replay_buffer.buffer) >= batch_size:
                loss = agent.optimize_model(replay_buffer, batch_size, beta)
        step_bar.close()
        agent.update_epsilon()
        
        # Update dynamic reward scaling
        env.dynamic_reward_scale = 1.0 + 0.01 * episode
        
        logger.info("Episode %d: Total Reward = %.2f, Epsilon = %.3f", episode+1, episode_reward, agent.epsilon)
        
        if (episode + 1) % target_update == 0:
            agent.update_target_network()
            logger.info("Updated target network at episode %d", episode+1)
    
    return agent
