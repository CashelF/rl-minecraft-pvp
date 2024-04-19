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

  return env

def preprocess_observation(info: dict) -> torch.Tensor:
  try:
    # Extract the observation
    observation = info['observation']

    # Add agent information
    feature = [observation['XPos'], observation['YPos'], observation['ZPos'], observation['Yaw'], observation['Life']]

    # Add hostile mob information
    for entity in observation['entities']:
      if entity['name'] == 'Zombie':
        feature.append(entity['x'])
        feature.append(entity['y'])
        feature.append(entity['z'])
        feature.append(entity['yaw'])
        feature.append(entity['life'])

    if len(feature) < FEATURE_SIZE:
      feature += [0] * (FEATURE_SIZE - len(feature))

    return torch.tensor(feature, dtype=torch.float32)
  except:
    return torch.zeros(FEATURE_SIZE, dtype=torch.float32)

def train(env, model: nn.Module, episodes: int = 500, gamma: float = 0.9, initial_epsilon: float = 0.9, final_epsilon: float = 0.1, epsilon_decay: float = 0.99, batch_size: int = 64, loss_fn: nn.Module = nn.MSELoss(), device: str = "cpu", log_dir: str = "logs/ResNet50", output_path: str = "models/model.pt"):

  # Set the model to training mode
  model.train()

  # Define the optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # Create a tensorboard writer
  writer = SummaryWriter(log_dir=log_dir)

  # Initialize a doubly ended queue to store training examples
  memory = deque(maxlen=10000)

  # Initializ epsilon
  epsilon = initial_epsilon
  
  # env.agent_host.sendCommand("use 1")
  # env.agent_host.sendCommand("hotbar.2")
  # env.agent_host.sendCommand("hotbar.1 0")
  # env.agent_host.sendCommand("hotbar.2 0")
  # env.agent_host.sendCommand("hotbar.3 1")
  # env.agent_host.sendCommand("hotbar.3 0")
  # env.agent_host.sendCommand("hotbar.4 1")
  # env.agent_host.sendCommand("hotbar.4 0")
  # env.agent_host.sendCommand("use 0")

  # Run the training loop
  for episode in range(episodes):

    # Reset the environment
    frame = env.reset()

    # First random step
    action = env.action_space.sample()

    # Step the environment
    frame, reward, done, info = env.step(action)

    state = preprocess_observation(info['observation'])

    episode_reward = 0
    episode_length = 0

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

            action = torch.argmax(model(state)).item()

            # Bring back from the device
            state = state.cpu()

        # Step the environment
        frame, reward, done, info = env.step(action)

        # Preprocess the next state
        next_state = preprocess_observation(info)
        
        # This is very scuffed. There must be a better way to do this. Too bad!
        try:
          if info['observation']['MobsKilled'] > 0:
            env.agent_host.sendCommand("quit")
            reward = 10
            done = True
        except:
          print("Too bad!")

        episode_reward += reward
        episode_length += 1

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


    print(f"Episode {episode + 1} Loss: {running_loss / len(memory):.2f} Average Reward: {episode_reward/episode_length:.2f}")

    # Log the loss
    writer.add_scalar("Loss", running_loss / len(memory), episode)

    # Log the epsilon
    writer.add_scalar("Epsilon", epsilon, episode)

    # Log the reward
    writer.add_scalar("Reward", episode_reward / episode_length, episode)

    # Decay epsilon
    epsilon = max(final_epsilon, epsilon_decay * epsilon)  

    # Save the model
    torch.save(model.state_dict(), output_path)
  env.close()



if __name__ == "__main__":
  # Initialize the environment
  env = initialize_environment()  

  # Determine the device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Load the model
  model = build_model(FEATURE_SIZE, env.action_space.n, device)

  # Train the model
  train(env, model, episodes=1000, batch_size=64, device=device, output_path="models/model.pt")

  