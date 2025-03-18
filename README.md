# CICIDS_2017_RL

## Overview

This project implements a complete, self-contained Reinforcement Learning environment for the CICIDS2017 dataset designed for network intrusion detection. The system uses a modular multi-agent reinforcement learning architecture where specialized agents work together to detect and classify different types of network attacks.

The implementation incorporates state-of-the-art techniques from research in reinforcement learning for cybersecurity, including temporal feature stacking, adversarial robustness, prioritized experience replay, dynamic reward scaling, and cost-sensitive penalties to handle class imbalance issues inherent in intrusion detection datasets.

## Features

- **DQN-based Implementation**: Deep Q-Network (DQN) implementation with important enhancements
- **Temporal Feature Stacking**: Combines consecutive time frames to capture temporal patterns in network traffic
- **Prioritized Experience Replay**: Optimizes learning by focusing on important transitions
- **Dynamic Reward Scaling**: Improves agent learning through adaptive reward mechanisms
- **Cost-Sensitive Penalties**: Addresses class imbalance by applying higher weights to minority classes
- **Single CSV File Operation**: Works with a preprocessed CSV file of the CICIDS2017 dataset
- **Adaptive to New Attacks**: Architecture designed to accommodate new attack types without retraining the entire system

## Requirements

The project requires the following dependencies:

```
numpy
pandas
torch
gym
scikit-learn
matplotlib
tqdm
```

## Project Structure

```
cicids_rl/
├── requirements.txt         # Required dependencies
├── data_loader.py           # Data loading and preprocessing module
├── custom_env.py            # Reinforcement learning environment (gym-compatible)
├── memory.py                # Prioritized experience replay implementation
├── agent.py                 # DQN agent implementation
├── training.py              # Training loop module
├── evaluation.py            # Evaluation metrics module
└── main.py                  # Main script that ties everything together
```

## Installation

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/WalkingDevFlag/cicids-rl.git
cd cicids-rl
```

2. Create a virtual environment (optional but recommended):

```bash
conda create -n cicids_rl python=3.10
conda activate cicids_rl
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure the preprocessed CICIDS2017 dataset (CSV file) is in the project directory
2. Run the main script:

```bash
python main.py
```

The script will automatically:
- Load and preprocess the CICIDS2017 dataset
- Initialize the RL environment and agents
- Train the agents using the specified parameters
- Evaluate the performance with appropriate metrics

## Architecture

The system consists of two main components:

1. **Data Preprocessing**: The raw CICIDS2017 dataset is transformed into a format suitable for reinforcement learning by:
   - Cleaning and normalizing network traffic features
   - Encoding attack labels
   - Creating temporal feature stacks to capture patterns over time

2. **Multi-Agent RL Framework**: 
   - **Level 1 Agents**: Multiple independent agents specialized in detecting specific attack types
   - **Decision-Maker Agent**: Combines outputs from Level 1 agents to make final classifications
   - **Gym-Compatible Environment**: Custom environment that implements the OpenAI Gym interface

## Key Components

### Data Loader

Handles preprocessing of the CICIDS2017 dataset:
- Loads the CSV file with network traffic data
- Normalizes features using StandardScaler
- Performs label encoding for attack types
- Implements temporal feature stacking to capture sequential patterns

### Custom Environment

A gym-compatible RL environment that:
- Defines state and action spaces for network traffic classification
- Implements a reward function with cost-sensitive penalties
- Provides step, reset, and render methods for agent interaction

### Memory Module

Implements prioritized experience replay for efficient learning:
- Stores (state, action, reward, next_state, done) tuples
- Assigns priorities to experiences based on their importance
- Samples batches based on priority for more effective learning

### DQN Agent

Implements the reinforcement learning algorithm:
- Deep Q-Network architecture with customizable hidden layers
- Epsilon-greedy action selection with annealing
- Weighted loss function to handle class imbalance
- Q-value updating using prioritized experiences

### Training Loop

Controls the training process:
- Handles episode progression and tracking
- Updates agent parameters and exploration rates
- Applies dynamic reward scaling
- Records performance metrics during training

### Evaluation Metrics

Comprehensive evaluation metrics including:
- Accuracy, precision, recall, and F1-score
- Confusion matrix analysis
- False positive rate tracking
- ROC curves and AUC measurement

## Performance

When properly trained, the model achieves:
- Overall accuracy of approximately x% on the CICIDS2017 dataset
- False positive rate of less than y%
- Effective detection of both common and rare attack types
- Robust handling of class imbalance issues

## License

This project is licensed under the MIT License

## Acknowledgments

This implementation is based on concepts from several research papers:
- "Multi-agent Reinforcement Learning-based Network Intrusion Detection System" [2407.05766]
- "Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey" [2405.20038]
- "DRL-IDS: An Improved Deep Reinforcement Learning Approach for Intrusion Detection Systems" [2111.13978]
- Additional research papers [2401.12262], [2401.05678], [2304.05822]
