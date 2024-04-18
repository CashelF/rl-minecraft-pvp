# $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000
import json
import random
import time
from collections import deque

import marlo
import numpy as np
import torch
import torch.nn as nn

# transforms
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm


def build_model(num_actions: int, device: str = "cpu"):
    # Load the model architecture
    model = resnet50()

    # Modify the last layer to output num_actions
    model.fc = torch.nn.Linear(model.fc.in_features, num_actions)

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

  return env

def train(env, model: nn.Module, transformations = nn.Identity, episodes: int = 500, gamma: float = 0.9, initial_epsilon: float = 0.5, final_epsilon: float = 0.1, epsilon_decay: float = 0.9, batch_size: int = 64, loss_fn: nn.Module = nn.MSELoss(), device: str = "cpu"):

  # Set the model to training mode
  model.train()

  # Define 
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # Initialize a doubly ended queue to store training examples
  memory = deque(maxlen=10000)

  # Initializ epsilon
  epsilon = initial_epsilon

  # Run the training loop
  for episode in range(episodes):

    # Reset the environment
    state = env.reset()

    # Apply transformations
    state = transformations(state)

    # Run the episode to completion
    done = False
    while not done:
        # Select an action
        if random.random() < epsilon: # Random action
            action = env.action_space.sample()
        else: # Action from the model
          with torch.no_grad():
            # Send the state to the device
            state = state.to(device)

            action = torch.argmax(model(state.unsqueeze(0))).item()

            # Bring back from the device
            state = state.cpu()

        # Step the environment
        next_state, reward, done, info = env.step(action)

        # Apply transformations
        next_state = transformations(next_state)

        # This is very scuffed. There must be a better way to do this. Too bad!
        if info and info['observation'] and info['observation']['MobsKilled'] > 0:
          env.agent_host.sendCommand("quit")
          reward = 100
          done = True

        if reward != 0:
           print(f"Reward: {reward}")

        # Store the transition
        memory.append((state, action, reward, next_state, done))

        # Update the state
        state = next_state

    running_loss = 0

    for states, actions, rewards, next_states, dones in tqdm(DataLoader(memory, batch_size, shuffle=True), desc=f"Episode {episode + 1} Training", unit="batch"):
        # Expand the actions, rewards, and dones
        states = states.float()
        actions = actions.long().unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        next_states = next_states.float()
        dones = dones.long().unsqueeze(1)

        # Send the batch to the device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # Compute the Q-values
        current_q_values = model(states).gather(1, actions)
        next_q_values = model(next_states).max(1)[0].detach().unsqueeze(1)
        expected_q_values = rewards + gamma * next_q_values * (1 - dones)

        # Ensure the expected_q_values tensor is float
        expected_q_values = expected_q_values.float()

        # Compute the loss
        loss = loss_fn(current_q_values, expected_q_values)

        # Tally the loss
        running_loss += loss.item()

        # Perform a gradient descent step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Bring back to the CPU
        states = states.cpu()
        actions = actions.cpu()
        rewards = rewards.cpu()
        next_states = next_states.cpu()
        dones = dones.cpu()

    # Decay epsilon
    epsilon = max(final_epsilon, epsilon_decay * epsilon)  

    print(f"Episode {episode + 1} Loss: {running_loss / len(memory)}")
  env.close()



if __name__ == "__main__":
  # Initialize the environment
  env = initialize_environment()  

  # Determine the device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Load the model
  model = build_model(env.action_space.n, device)

  # Define image transformations
  transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((150, 200))
  ])

  # Train the model
  train(env, model, transformations, episodes=100, batch_size=64, device=device)


  