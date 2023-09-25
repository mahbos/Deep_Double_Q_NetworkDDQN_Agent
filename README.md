Deep Double Q-Network (DDQN) Agent for the KukaBullet Environment in OpenAI's Gym

This script demonstrates the use of a Deep Double Q-Network (DDQN) agent to learn and navigate the KukaBullet environment provided by OpenAI's Gym and PyBullet.

Key Features:

Environment Setup:
Uses the KukaBulletEnv-v0 environment from Gym.
Defined possible discrete actions for the robot.
DDQNAgent:
Architecture: 2-hidden-layer dense neural network.
Replay Memory: Stores experiences to later sample and train the network.
Epsilon-greedy Strategy: Implements exploration and exploitation.
Target Network: Used for stabilizing Q-learning updates.
Training Process:
The agent interacts with the environment for a fixed number of episodes.
The agent's actions result from either random exploration (based on epsilon) or the current policy (network's highest Q-value prediction).
Stores experiences in memory, and after certain steps, samples from this memory to train the network.
Network Updates:
Uses the target model to compute the Q-values, stabilizing the training process.
Epsilon decays over time to reduce the exploration and rely more on the policy learned.
Simulation Control:
Integrates directly with the PyBullet physics server.
Can connect to either DIRECT mode for faster computation or GUI mode for visual feedback.
