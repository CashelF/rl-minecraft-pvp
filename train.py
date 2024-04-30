import datetime
import glob
import os
import pickle
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import FEATURE_SIZE, NUM_ACTIONS, build_model, load_trajectory_data


def train(model: nn.Module, trajectories: DataLoader, num_epochs: int = 10, gamma: float = 0.9, loss_fn: nn.Module = nn.MSELoss(), log_dir: str = "logs/", model_dir: str = "models/", device: str = "cpu"):
    """
    Train the model using pre-collected trajectories stored in memory.
    
    Args:
    - model: The PyTorch model to be trained.
    - trajectories: Dataloader containing pre-collected trajectories.
    - num_epochs: Number of epochs.
    - gamma: Discount factor for future rewards.
    - loss_fn: Loss function used for evaluating performance.
    - device: The device (CPU or GPU) that training will occur on.
    """
    # Set the model to training mode
    model.train()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0

    # Train the model
    for epoch in tqdm(range(num_epochs), desc="Training on Trajectories", unit="Epoch"):
        for states, actions, rewards, next_states, dones in trajectories:
            # Expand the actions, rewards, and dones
            actions = actions.long().unsqueeze(1)
            rewards = rewards.unsqueeze(1)
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
            loss: torch.Tensor = loss_fn(current_q_values, expected_q_values)

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

            # Log the loss to tensorboard
            writer.add_scalar("Loss", loss.item(), global_step)

            global_step += 1
        
        torch.save(model.state_dict(), os.path.join(model_dir, f"model{timestamp}.pt"))


def train_random_sample(model: nn.Module, memory: deque, num_batches: int, gamma: float = 0.9, batch_size: int = 64, loss_fn: nn.Module = nn.MSELoss(reduction='none'), device: str = "cpu") -> float:
    """Train the model by randomly sampling from the experience replay buffer. Returns the average loss accross all batches"""
    # Set the model to training mode
    model.train()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    running_loss = 0

    for i in range(num_batches):
        # Sample a batch from the memory
        batch = random.sample(memory, min(batch_size, len(memory)))

        # Unpack the batch
        states, actions, rewards, next_states, dones, behavior_probabilities = zip(*batch)

        # Expand the actions, rewards, and dones
        states = torch.stack(states).float()
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards).unsqueeze(1)
        next_states = torch.stack(next_states).float()
        dones = torch.tensor(dones, dtype=torch.long).unsqueeze(1)
        behavior_probabilities = torch.tensor(behavior_probabilities).unsqueeze(1)

        # Send the batch to the device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        behavior_probabilities = behavior_probabilities.to(device)


        # Compute the Q-values
        current_q_values = model(states)

        # Calculate the importance sampling ratio
        target_probabilities = torch.softmax(current_q_values, dim=1).gather(1, actions)

        importance_sampling_ratio = target_probabilities / behavior_probabilities

        # Compute the Q-values
        current_q_values = current_q_values.gather(1, actions)
        next_q_values = model(next_states).max(1)[0].detach().unsqueeze(1)
        expected_q_values = rewards + gamma * next_q_values * (1 - dones)

        # Ensure the expected_q_values tensor is float
        expected_q_values = expected_q_values.float()

        # Calculate the TD error
        loss: torch.Tensor = torch.mean((importance_sampling_ratio * (expected_q_values - current_q_values)).pow(2))

        # loss: torch.Tensor = torch.mean((importance_sampling_ratio * td_error).pow(2))

        # Compute the loss
        # loss: torch.Tensor = torch.mean(importance_sampling_ratio * loss_fn(current_q_values, expected_q_values))

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
        behavior_probabilities = behavior_probabilities.cpu()

    return running_loss / num_batches

# Usage example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(torch.__version__)
    # print(torch.version.cuda)
    # print(f"Device: {device}")
    
    # Build the model
    model = build_model(FEATURE_SIZE, NUM_ACTIONS)

    print("Loading trajectories...")
    
    # Load the trajectory data
    trajectories = load_trajectory_data("trajectories/trajectory_data_2024-04-30_02-04-57.pkl")

    print(f"{len(trajectories)} Trajectories loaded.")

    # Train the model
    train(model, DataLoader(trajectories, batch_size=131072), num_epochs=40, log_dir="logs/SmallModel D=0.4", device=device)