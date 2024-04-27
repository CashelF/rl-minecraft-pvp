import math
import pickle
import random
from collections import deque

import marlo
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

FEATURE_SIZE = 3
HIDDEN_SIZE = 256
DROPOUT_PROB = 0.1


def save_data_with_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

<<<<<<< HEAD

def build_model(num_inputs: int, num_actions: int, device: str = "cpu") -> nn.Module:
=======
def build_model(num_inputs: int, num_actions: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
>>>>>>> 5a94b5a1b1ff0693a8960b7408e6a86c297b68d2
    model = nn.Sequential(
        nn.Linear(num_inputs, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(DROPOUT_PROB),
        nn.Linear(num_inputs, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(DROPOUT_PROB),
        nn.Linear(HIDDEN_SIZE, num_actions),
    )

    return model.to(device)


def preprocess_observation(info: dict):
    try:
        observation = info["observation"]

        x, z, yaw, life = observation["XPos"], observation["ZPos"], observation["Yaw"], observation["Life"]

        # Bound yaw between -180 and 180
        yaw = yaw + 360 if yaw < -180 else yaw - 360 if yaw > 180 else yaw

        
        for entity in observation["entities"]:
            if entity["name"] == "Zombie":
                dx, dz = entity["x"] - x, entity["z"] - z
                distance_to_mob = math.sqrt(dx**2 + dz**2)
                yaw_to_mob = -180 * math.atan2(dx, dz) / math.pi
                return torch.tensor(
                    [distance_to_mob, yaw_to_mob, yaw], dtype=torch.float32
                )
        return torch.zeros(FEATURE_SIZE, dtype=torch.float32)
    except KeyError:
        return torch.zeros(FEATURE_SIZE, dtype=torch.float32)

<<<<<<< HEAD

def train(env, model: nn.Module, episodes: int = 100, device: str = "cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Replace with a better policy
            next_state, reward, done, _ = env.step(action)
            loss = loss_fn(
                model(state), model(next_state)
            )  # Simplified loss calculation
=======
@marlo.threaded
def agent_thread(join_token, model, can_train=False):
    env = marlo.init(join_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(env, model, episodes=100, device=device, can_train=can_train)
    env.close()

def train(env, model: nn.Module, episodes: int = 500, gamma: float = 0.9, initial_epsilon: float = 0.9, final_epsilon: float = 0.1, epsilon_decay: float = 0.995, batch_size: int = 64, loss_fn: nn.Module = nn.MSELoss(), device: str = "cpu", log_dir: str = "logs/", output_path: str = "models/model.pt", can_train=False):

  # Define the optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  # Create a tensorboard writer
  writer = SummaryWriter(log_dir=log_dir)

  # Initialize a doubly ended queue to store training examples
  memory = deque(maxlen=50000)

  # Initializ epsilon
  epsilon = initial_epsilon
  
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

    state = preprocess_observation(info['observation'])

    episode_reward = 0
    episode_length = 0

    # Run the episode to completion
    done = False
    while not done:
        # TODO: implement this >
        # if env.agent_host.peekWorldState().is_mission_running == False:
        #   break
        
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

        reward -= abs(next_state[1] - next_state[2]).item() / 180
        
        # This is very scuffed. There must be a better way to do this. Too bad!
        try:
          if info['observation']['MobsKilled'] > 0:
            print("You killed the zombie!")
            env.agent_host.sendCommand("quit")
            reward = 100
            done = True
        except:
          print("Too bad!")

        episode_reward += reward
        episode_length += 1

        # if reward != 0:
        #    print(f"Reward: {reward}")

        # Store the transition
        memory.append((state, action, reward, next_state, done))

        # Update the state
        state = next_state

    if can_train:
    
        # Set the model to training mode
        model.train()

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
>>>>>>> 5a94b5a1b1ff0693a8960b7408e6a86c297b68d2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Bring back to the CPU
            states = states.cpu()
            actions = actions.cpu()
            rewards = rewards.cpu()
            next_states = next_states.cpu()
            dones = dones.cpu()


    print(f"Episode {episode + 1} Loss: {running_loss / len(memory):.2f} Episode Reward: {episode_reward:.2f}")

    # Log the loss
    writer.add_scalar("Loss", running_loss / len(memory), episode)

    # Log the epsilon
    writer.add_scalar("Epsilon", epsilon, episode)

    # Log the reward
    writer.add_scalar("Reward", episode_reward, episode)

    # Log the episode length
    writer.add_scalar("Episode Length", episode_length, episode)

    # Decay epsilon
    epsilon = max(final_epsilon, epsilon_decay * epsilon)  

    # Save the model
    torch.save(model.state_dict(), output_path)
    
    save_data_with_pickle(memory, f"trajectories/trajectory_data_episode_{episode}.pkl")
    
  env.close()

@marlo.threaded
def start_agent(join_token, model: nn.Module, device="cpu"):
    # Initialize the environment
    env = marlo.init(join_token)

    # Train for 100 episodes
    train(env, model, episodes=100, device=device)

    # Close the environment
    env.close()


if __name__ == "__main__":
    client_pool = [("127.0.0.1", 10000), ("127.0.0.1", 10001)]
    join_tokens = marlo.make(
        "mission.xml",
        params={
            "client_pool": client_pool,
            "suppress_info": False,
            "kill_clients_after_num_rounds": 1000,
            "videoResolution": [800, 600],
            "PrioritiesOffscreenRendering": False,
        },
    )

    # Ensure we have two agents
    assert len(join_tokens) == 2
    
    model = build_model(FEATURE_SIZE, 9)

<<<<<<< HEAD
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
=======
    thread_handler_0, _ = agent_thread(join_tokens[0], model, can_train=True)
    thread_handler_1, _ = agent_thread(join_tokens[1], model, can_train=False)
    thread_handler_0.join()
    thread_handler_1.join()
    
    
>>>>>>> 5a94b5a1b1ff0693a8960b7408e6a86c297b68d2

    # Build the model
    model = build_model(FEATURE_SIZE, 9, 6, device=device)

    print("Starting Agents")

    threads = []

    threads.append(start_agent(join_tokens[0])[0])
    threads.append(start_agent(join_tokens[1])[0])

    # Wait for training to finish
    for thread in threads:
        thread.join()

    print("Training Complete")
