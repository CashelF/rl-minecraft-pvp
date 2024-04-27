import math
import pickle
import random
from collections import deque

import marlo
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

FEATURE_SIZE = 3
HIDDEN_SIZE = 256
DROPOUT_PROB = 0.1


def save_data_with_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def build_model(num_inputs: int, num_actions: int, device: str = "cpu") -> nn.Module:
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Episode {episode + 1}: Loss = {loss.item()}")

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

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
