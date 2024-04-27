import math
import os
import pickle
import random
import threading
from collections import deque

import marlo
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

FEATURE_SIZE = 7
HIDDEN_SIZE = 256
DROPOUT_PROB = 0.4


def save_data_with_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def build_model(num_inputs: int, num_actions: int):
    model = nn.Sequential(
        nn.Linear(num_inputs, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(DROPOUT_PROB),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(DROPOUT_PROB),
        nn.Linear(HIDDEN_SIZE, num_actions),
    )

    return model


def preprocess_observation(info):
    try:
        observation = info["observation"]

        x, z, yaw, life = (
            observation["XPos"],
            observation["ZPos"],
            observation["Yaw"],
            observation["Life"],
        )

        # Bound yaw between -180 and 180
        yaw = yaw + 360 if yaw < -180 else yaw - 360 if yaw > 180 else yaw

        for entity in observation["entities"]:
            if entity["name"] != observation["Name"]:
                # Extract the enemy's position and life
                enemy_x, enemy_z, enemy_life = entity["x"], entity["z"], entity["life"]

                # Calculate the distance & angle to the enemy
                dx, dz = enemy_x - x, enemy_z - z

                distance_to_enemy = math.sqrt(dx**2 + dz**2)

                yaw_to_enemy = -180 * math.atan2(dx, dz) / math.pi

                features = torch.tensor(
                    [x, z, yaw, life, distance_to_enemy, yaw_to_enemy, enemy_life],
                    dtype=torch.float32,
                )

                return features

        return torch.zeros(FEATURE_SIZE, dtype=torch.float32)
    except:
        return torch.zeros(FEATURE_SIZE, dtype=torch.float32)


def train(
    env,
    memory: deque,
    model: nn.Module,
    episodes: int = 500,
    gamma: float = 0.9,
    initial_epsilon: float = 0.9,
    final_epsilon: float = 0.1,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    loss_fn: nn.Module = nn.MSELoss(),
    log_dir: str = "logs/",
    model_dir: str = "models/",
    trajectory_dir: str = "trajectories/",
    can_train: bool = False,
):
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create a tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Ensure the output directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(trajectory_dir, exist_ok=True)

    # Initialize epsilon
    epsilon = initial_epsilon

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

        episode_reward = 0
        episode_length = 0

        # Run the episode to completion
        done = False
        while not done:
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

            # Step the environment
            frame, reward, done, info = env.step(action)

            # Preprocess the next state
            next_state = preprocess_observation(info)

            # This is very scuffed. There must be a better way to do this. Too bad!
            try:
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

            # Update the state
            state = next_state

        print(f"Episode Reward({threading.current_thread()}): {episode_reward:.2f}")

        if can_train:
            # Set the model to training mode
            model.train()

            running_loss = 0

            for i in range(episode_length):
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

            # Log the loss
            writer.add_scalar("Loss", running_loss / episode_length, episode)

            # Log the epsilon
            writer.add_scalar("Epsilon", epsilon, episode)

            # Log the reward
            writer.add_scalar("Reward", episode_reward, episode)

            # Log the episode length
            writer.add_scalar("Episode Length", episode_length, episode)

            # Decay epsilon
            epsilon = max(final_epsilon, epsilon_decay * epsilon)

            # Save the model
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

            # Save the memory
            save_data_with_pickle(
                memory,
                os.path.join(trajectory_dir, f"trajectory_data_episode_{episode}.pkl"),
            )


@marlo.threaded
def start_agent(join_token, memory, model: nn.Module, can_train: bool = False):
    # Initialize the environment
    env = marlo.init(join_token)

    # Train for 100 episodes
    train(env, memory, model, episodes=100, can_train=can_train)

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
            "videoResolution": [800, 600],
            "PrioritiesOffscreenRendering": False,
        },
    )

    # Ensure we have two agents
    assert len(join_tokens) == 2

    # Build the model
    model = build_model(FEATURE_SIZE, 9)

    # Initialize the memory
    memory = deque(maxlen=10000)

    print("Starting Agents")

    threads = []

    threads.append(start_agent(join_tokens[0], memory, model, can_train=True)[0])
    threads.append(start_agent(join_tokens[1], memory, model)[0])

    # Wait for training to finish
    for thread in threads:
        thread.join()

    print("Training Complete")