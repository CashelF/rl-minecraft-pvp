import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
from tqdm import tqdm
import pickle
import os
import glob
import datetime

FEATURE_SIZE = 7
NUM_ACTIONS = 9
HIDDEN_SIZE = 128

def load_trajectory_data(filename):
    """Load the entire list of trajectory data from a pickle file."""
    with open(filename, 'rb') as file:
        trajectory = pickle.load(file)
    return trajectory

def build_model(num_inputs: int, num_actions: int):
    model = nn.Sequential(
        nn.Linear(num_inputs, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, num_actions),
    )

    return model
    

def train_network(model, memory, episodes, batch_size, gamma, optimizer, loss_fn, device, trajectory_directory, log_dir: str = "logs/", model_dir: str = "models/"):
    """
    Train the model using pre-collected trajectories stored in memory.
    
    Args:
    - model: The PyTorch model to be trained.
    - memory: Deque or list containing pre-collected trajectories.
    - episodes: Number of episodes (full passes over the memory) for training.
    - batch_size: Size of the batch to use for training.
    - gamma: Discount factor for future rewards.
    - optimizer: Optimizer used for training the model.
    - loss_fn: Loss function used for evaluating performance.
    - device: The device (CPU or GPU) that training will occur on.
    """
    trajectory_files = glob.glob(os.path.join(trajectory_directory, '*.pkl'))
    print(f"Found {len(trajectory_files)} files.")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    model.to(device)
    model.train()
    for file_path in trajectory_files:
        episode_data = load_trajectory_data(file_path)
        memory.extend(episode_data)
        running_loss = 0
        episode_length = len(episode_data)

        for i in tqdm(range(len(memory) // batch_size)):
            # Sample a batch from the memory
            batch = random.sample(memory, min(batch_size, len(memory)))

            # Unpack the batch
            states, actions, rewards, next_states, dones = zip(*batch)

            # Expand the actions, rewards, and dones
            states = torch.stack(states).float()
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(rewards).unsqueeze(1)
            next_states = torch.stack(next_states).float()
            dones = torch.tensor(dones, dtype=torch.long).unsqueeze(1)

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

        # Save the model
        torch.save(model.state_dict(), os.path.join(model_dir, f"model{timestamp}.pt"))

# Usage example:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print(torch.version.cuda)
print("Device: ", device)
model = build_model(FEATURE_SIZE, NUM_ACTIONS)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
batch_size = 64
gamma = 0.99
episodes = 100  # Number of training passes over the data

memory = deque(maxlen=50000)

# Now, call the training function:
train_network(model, memory, episodes, batch_size, gamma, optimizer, loss_fn, device, "trajectories/")