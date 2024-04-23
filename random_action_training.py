# $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000
import random
from collections import deque

import marlo
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from environment import ZombieEnv
from replay_buffer import ReplayBuffer

FEATURE_SIZE = 10
HIDDEN_SIZE = 256
DROPOUT_PROB = 0.1

def build_model(num_inputs: int, num_actions: int, device: str = "cpu"):
  # Fully connected model
  model = nn.Sequential(
    nn.Linear(num_inputs, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Dropout(DROPOUT_PROB),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Dropout(DROPOUT_PROB),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Dropout(DROPOUT_PROB),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Dropout(DROPOUT_PROB),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Dropout(DROPOUT_PROB),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Dropout(DROPOUT_PROB),
    nn.Linear(HIDDEN_SIZE, num_actions)
  )

  # Send the model to the device
  model = model.to(device)

  return model

def initialize_environment(mission_file: str = "mission.xml"):
  # Define the client pool
  client_pool = [('127.0.0.1', 10000)]

  # Create the environment. Whoever made suppress_info default to False, I will find you.
  join_tokens = marlo.make(mission_file, params={"client_pool": client_pool, "suppress_info": False})

  # Initialize the environment (Assuming only one agent)
  env = marlo.init(join_tokens[0])
  env = ZombieEnv(env)
  return env

def collect_data(env, episodes):
    data = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Random action
            action = 9 if random.random() < 0.1 else action # 10% of the time, look at the zombie
            next_state, reward, done, info = env.step(action)
            data.append((state, action, reward, next_state, done))
            state = next_state
    return data

import torch
import torch.nn.functional as F

def update_model(model, experiences, optimizer, gamma=0.99):
    states, actions, rewards, next_states, dones = experiences

    # Convert numpy arrays to torch tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Get current Q values from model
    current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute the maximum Q values at the next states from the target model
    next_q_values = model(next_states).max(1)[0]
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    # Compute loss
    loss = F.mse_loss(current_q_values, target_q_values)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_model(model, data, batch_size):
    replay_buffer = ReplayBuffer(capacity=100000)
    replay_buffer.store_many(data)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    batch_size = 64
    num_training_steps = 1000

    for _ in range(num_training_steps):
        if len(replay_buffer) < batch_size:
            continue
        experiences = replay_buffer.sample(batch_size)
        loss = update_model(model, experiences, optimizer)
        print(f"Training Loss: {loss}")

# Example usage
env = initialize_environment()
data = collect_data(env, 10000)  # Collect data from 10,000 episodes
model = build_model()
train_model(model, data, 64)  # Train model using the collected data
