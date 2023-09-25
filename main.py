# Install required packages
!pip install gym pyvirtualdisplay > /dev/null 2>&1
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
!apt-get install python-opengl -y
!apt install xvfb -y
!pip install pyvirtualdisplay
!pip install piglet
!pip install gym pybullet keras

import gym
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
import os
import io
import base64
from IPython.display import HTML
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pybullet as p

# Connect to physics server
#p.connect(p.DIRECT)  # Or p.GUI if you want to see the simulation visually

# Ensure you have the required packages:
# pip install gym[all] pybullet

# Define possible actions for the KukaBullet environment.
POSSIBLE_ACTIONS = [
    [-1, 0, 0], [1, 0, 0],
    [0, -1, 0], [0, 1, 0],
    [0, 0, -1], [0, 0, 1]
]

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, len(POSSIBLE_ACTIONS) - 1)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


env = gym.make('KukaBulletEnv-v0')


# Add this line right after you create the environment
p.connect(p.DIRECT)

# ... [rest of your code]

state_size = env.observation_space.shape[0]
action_size = len(POSSIBLE_ACTIONS)

agent = DDQNAgent(state_size, action_size)

EPISODES = 100
batch_size = 32

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        #env.render()
        action_index = agent.act(state)
        action = POSSIBLE_ACTIONS[action_index]
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action_index, reward, next_state, done)
        state = next_state
        if done:
            agent.update_target_model()
            print(f"episode: {e}/{EPISODES}, score: {time}, e: {agent.epsilon:.2f}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

env.close()
