# custom_env.py
import gym
from gym import spaces
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CIDS2017Env(gym.Env):
    """
    Gym environment for the CICIDS2017 intrusion detection dataset.
    The environment iterates through the dataset samples (states) and expects
    an action corresponding to the predicted class.
    It includes dynamic reward scaling and cost-sensitive penalties.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, X, y, dynamic_reward_scale=1.0, cost_penalty=2.0):
        super(CIDS2017Env, self).__init__()
        self.X = X
        self.y = y
        self.num_samples = X.shape[0]
        self.current_index = 0
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.action_space = spaces.Discrete(self.num_classes)
        
        self.dynamic_reward_scale = dynamic_reward_scale
        self.cost_penalty = cost_penalty

    def reset(self):
        """Resets the environment and returns the first state."""
        self.current_index = 0
        return self.X[self.current_index]

    def step(self, action):
        """
        Execute one step in the environment.
        
        Returns:
            next_state (np.array), reward (float), done (bool), info (dict)
        """
        true_label = self.y[self.current_index]
        # Reward: correct gives positive; wrong gets a penalty.
        if action == true_label:
            reward = 1.0 * self.dynamic_reward_scale
        else:
            reward = -self.cost_penalty * self.dynamic_reward_scale

        info = {"true_label": true_label}
        self.current_index += 1
        done = self.current_index >= self.num_samples
        
        if not done:
            next_state = self.X[self.current_index]
        else:
            next_state = np.zeros(self.X.shape[1], dtype=np.float32)
        
        return next_state, reward, done, info

    def render(self, mode='human', close=False):
        logger.info("Step: %d / %d", self.current_index, self.num_samples)
