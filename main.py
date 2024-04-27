import math
import random
import pickle
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
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def build_model(num_inputs: int, num_actions: int, device: str = "cpu"):
    model = nn.Sequential(
        nn.Linear(num_inputs, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(DROPOUT_PROB),
        nn.Linear(HIDDEN_SIZE, num_actions)
    )
    return model.to(device)

def preprocess_observation(info: dict):
    try:
        observation = info['observation']
        x, z, yaw = observation['XPos'], observation['ZPos'], observation['Yaw']
        yaw = yaw + 360 if yaw < -180 else yaw - 360 if yaw > 180 else yaw
        for entity in observation['entities']:
            if entity['name'] == 'Zombie':
                dx, dz = entity['x'] - x, entity['z'] - z
                distance_to_mob = math.sqrt(dx ** 2 + dz ** 2)
                yaw_to_mob = -180 * math.atan2(dx, dz) / math.pi
                return torch.tensor([distance_to_mob, yaw_to_mob, yaw], dtype=torch.float32)
        return torch.zeros(FEATURE_SIZE, dtype=torch.float32)
    except KeyError:
        return torch.zeros(FEATURE_SIZE, dtype=torch.float32)

@marlo.threaded
def agent_thread(join_token):
    env = marlo.init(join_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(FEATURE_SIZE, env.action_space.n, device)
    train(env, model, episodes=100, device=device)
    env.close()

def train(env, model, episodes=100, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Replace with a better policy
            next_state, reward, done, _ = env.step(action)
            loss = loss_fn(model(state), model(next_state))  # Simplified loss calculation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Episode {episode + 1}: Loss = {loss.item()}")

if __name__ == "__main__":
    client_pool = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
    join_tokens = marlo.make('mission.xml', params={"client_pool": client_pool, "suppress_info": False, 'kill_clients_after_num_rounds': 1000, 'videoResolution': [800,600], "PrioritiesOffscreenRendering":False})
    assert len(join_tokens) == 2

    thread_handler_0, _ = agent_thread(join_tokens[0])
    thread_handler_0.join()
    print("Agent 0 Complete")
    thread_handler_1, _ = agent_thread(join_tokens[1])
    thread_handler_1.join()

    print("Episode Run Complete")
