import os
import random
import threading
from collections import deque

import marlo
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import (
    FEATURE_SIZE,
    build_model,
    get_new_filename,
    load_trajectory_data,
    preprocess_observation,
    save_trajectory,
)

trajectory_file_name = get_new_filename()


def train(model: nn.Module, trajectories: DataLoader, gamma: float, num_epochs: int = 10, loss_fn: nn.Module = nn.MSELoss(), device: str = "cpu"):
    """Train the model on trajectory data"""
    # Set the model to training mode
    model.train()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in tqdm(range(num_epochs), desc="Training on Trajectories", unit="Epoch"):
        for states, actions, rewards, next_states, dones in trajectories:
            # Expand the actions, rewards, and dones
            states = torch.tensor(states)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(rewards).unsqueeze(1)
            next_states = torch.stack(next_states)
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

def train_random_sample(model: nn.Module, memory: deque, num_batches: int, gamma: float, batch_size: int = 64, loss_fn: nn.Module = nn.MSELoss(), device: str = "cpu") -> float:
    """Train the model by randomly sampling from the experience replay buffer. Returns the average loss accross all batches"""
    # Set the model to training mode
    model.train()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    running_loss = 0

    for i in tqdm(range(num_batches), desc="Training on Experience Replay Buffer", unit="Batch"):
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

    return running_loss / num_batches


def play_episodes(
    env,
    model: nn.Module,
    memory: deque,
    trajectories: list,
    episodes: int = 500,
    gamma: float = 0.9,
    initial_epsilon: float = 0.9,
    final_epsilon: float = 0.1,
    epsilon_decay: float = 0.995,
    log_dir: str = "logs/Bootstrapping",
    model_dir: str = "models/",
    trajectory_dir: str = "trajectories/",
    can_train: bool = False,
):

    # Ensure the output directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(trajectory_dir, exist_ok=True)

    # Initialize epsilon
    epsilon = initial_epsilon

    if can_train:
        # Create a tensorboard writer
        writer = SummaryWriter(log_dir=log_dir)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send the model to the device
    model.to(device)

    # Run the training loop
    for episode in range(episodes):
        # Set the model to evaluation mode
        model.eval()

        # Reset the environment
        frame = env.reset()

        # First random step
        action = env.action_space.sample()

        # Step the environment
        frame, reward, done, info = env.step(action)

        state = preprocess_observation(info)
        
        current_damage_dealt = 0
        current_damage_taken = 0

        episode_reward = 0
        episode_length = 0

        # Run the episode to completion
        done = False
        while not done:
            if can_train: # Epsilon-Greedy Model Policy
                # Select an action
                if random.random() < epsilon:  # Random action
                    action = env.action_space.sample()
                else:  # Action from the model
                    with torch.no_grad():
                        # Send the state to the device
                        state = state.to(device)

                        action = torch.argmax(model(state)).item()

                        # Bring back from the device
                        state = state.cpu()
            else: # Hardcoded Policy
                x, z, yaw, life, distance_to_enemy, yaw_delta, enemy_life = state

                if abs(yaw_delta) < 5:
                    if distance_to_enemy < 3:
                        action = 5 if action == 0 else 0
                    else:
                        action = 1 if action not in [3, 4] else 0
                elif yaw_delta < 0: # Turn right
                    action = 3
                elif yaw_delta > 0: # Turn left
                    action = 4

            # Step the environment
            frame, reward, done, info = env.step(action)

            # Preprocess the next state
            next_state = preprocess_observation(info)

            # This is very scuffed. There must be a better way to do this. Too bad!
            try:
                if info["observation"]["DamageDealt"] > current_damage_dealt:
                    reward += (info["observation"]["DamageDealt"] - current_damage_dealt) / 10
                    current_damage_dealt = info["observation"]["DamageDealt"]
                    
                if info["observation"]["DamageTaken"] > current_damage_taken:
                    reward -= (info["observation"]["DamageTaken"] - current_damage_taken) / 10
                    current_damage_taken = info["observation"]["DamageTaken"]
                
                if info["observation"]["PlayersKilled"] > 0:
                    print(f"{info['observation']['Name']} killed the enemy!")
                    env.agent_host.sendCommand("quit")
                    reward = 100
                    done = True         
            except:
                print("Too bad!")

            episode_reward += reward
            episode_length += 1

            # Store the transition
            memory.append((state, action, reward, next_state, done))
            
            # Add trajectory
            trajectories.append((state, action, reward, next_state, done))

            # Update the state
            state = next_state

        print(f"Episode {episode} Reward({threading.current_thread()}): {episode_reward:.2f}")

        if can_train:
            # Train on trajectory data from the experience replay buffer
            loss = train_random_sample(model, memory, episode_length, gamma, device=device)

            # Log the loss
            writer.add_scalar("Loss", loss, episode)

            # Log the episode reward
            writer.add_scalar("Reward", episode_reward, episode)

            # Log the episode length
            writer.add_scalar("Episode Length", episode_length, episode)

            # Log the epsilon
            writer.add_scalar("Epsilon", epsilon, episode)

            # Decay epsilon
            epsilon = max(final_epsilon, epsilon_decay * epsilon)
            
            # Save the trajectory data
            save_trajectory(os.path.join(trajectory_dir, trajectory_file_name), trajectories)

            # Save the model
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))    


@marlo.threaded
def start_agent(join_token, memory, trajectories, model: nn.Module, can_train: bool = False):
    # Initialize the environment
    env = marlo.init(join_token)

    # Train for 2000 episodes
    play_episodes(env, memory, trajectories, model, episodes=4000, can_train=can_train)

    # Close the environment
    env.close()



if __name__ == "__main__":
    # Define the client pool
    client_pool = [("127.0.0.1", 10000), ("127.0.0.1", 10001)]

    # Create the environment
    join_tokens = marlo.make(
        "mission.xml",
        params={
            "client_pool": client_pool,
            "suppress_info": False,
            "kill_clients_after_num_rounds": 9999,
            "videoResolution": [400, 300],
            "PrioritiesOffscreenRendering": False,
        },
    )

    # Ensure we have two agents
    assert len(join_tokens) == 2

    # Build the model
    model = build_model(num_inputs=FEATURE_SIZE, num_actions=7)

    model.load_state_dict(torch.load("models/bot.pt"))

    # Initialize the memory
    memory = deque(maxlen=50000)
    
    trajectories = []

    print("Starting Agents")

    threads = []

    threads.append(start_agent(join_tokens[0], model, memory, trajectories, can_train=True)[0])
    threads.append(start_agent(join_tokens[1], model, memory, trajectories)[0])

    # Wait for training to finish
    for thread in threads:
        thread.join()

    print("Training Complete")