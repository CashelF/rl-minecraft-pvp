import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Adjust the flattened size according to your input shape
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        output = self.fc2(x)
        return output


def train_dqn(env):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    action_size = env.action_space.n
    state_size = (84, 84, 3)  # Assuming 84x84 RGB images
    model = DQN(state_size, action_size)

    # Send the model to the appropriate device
    model.to(device)
    
    episodes = 1000
    max_steps = 100
    batch_size = 32
    memory = deque(maxlen=2000)
    gamma = 0.95  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995

    for e in range(episodes):
        state = env.reset()
        state = np.array(state) / 255.0  # Normalize the image data to [0, 1]
        total_reward = 0

        for step in range(max_steps):
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                q_values = model(torch.tensor(state))
                action = np.argmax(q_values[0])

            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state) / 255.0
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                for s, a, r, s_next, d in minibatch:
                    target = r
                    if not d:
                        target = r + gamma * np.amax(model.predict(s_next[None, :, :, :])[0])
                    target_f = model.predict(s[None, :, :, :])
                    target_f[0][a] = target
                    model.train_on_batch(s[None, :, :, :], target_f)

            if done:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode: {e+1}/{episodes}, Reward: {total_reward}, Epsilon: {epsilon}")

    return model
