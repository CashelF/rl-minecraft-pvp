import datetime
import math
import pickle
import random
import time
from glob import glob

import torch
import torch.nn as nn

FEATURE_SIZE = 7
NUM_ACTIONS = 7

def get_new_filename() -> str:
    """Generate a unique filename using the current timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"trajectory_data_{timestamp}.pkl"

def save_trajectory(filename: str, trajectory) -> None:
    """Save the complete trajectory list to a binary file using pickle."""
    with open(filename, 'wb') as file:  # 'wb' for writing in binary
        pickle.dump(trajectory, file)

def load_trajectory_data(filename: str):
    """Load the entire list of trajectory data from a pickle file."""
    with open(filename, 'rb') as file:
        trajectory = pickle.load(file)
    return trajectory

def load_trajectory_directory(directory: str):
    """Load all the trajectory data from a directory of pickle files."""
    trajectories = []
    for file in glob(f"{directory}/*.pkl"):
        trajectories.extend(load_trajectory_data(file))
    return trajectories

def bound_angle(angle: float) -> float:
    """Bound an angle between -180 and 180 degrees."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360

    return angle

def build_model(num_inputs: int, num_actions: int, hidden_size: int = 512, dropout_prob: float = 0.2) -> nn.Module:
    """Build a simple fully connected network."""
    model = nn.Sequential(
        nn.Linear(num_inputs, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Linear(hidden_size, num_actions),
    )

    return model

def randomly_move_agent(env):
    """Move the agent randomly at the start of the episode to simulate changing spawn points."""
    # Turn randomly
    turn_command = f"turn {random.uniform(-1, 1):.2f}"
    env.agent_host.sendCommand(turn_command)
    time.sleep(2)  # Sleep to ensure the command is executed before the next one

    # Stop turning
    env.agent_host.sendCommand("turn 0")

    # Move randomly
    move_command = f"move {random.uniform(-1, 1):.2f}"
    env.agent_host.sendCommand(move_command)
    time.sleep(2)

    # Stop moving
    env.agent_host.sendCommand("move 0")
    

def encode_state(info: dict, previous_agent_yaw: float = 0.0):
    """Extract the hand-crafted state vector from an observation."""
    try:
        observation = info["observation"]

        # Find the agent and enemy entities
        agent = None
        enemy = None

        for entity in observation["entities"]:
            if entity["name"] == observation["Name"]:
                agent = entity
            elif entity["name"] != observation["Name"]:
                enemy = entity

        # If the agent or enemy is not found, return a zero vector
        if agent is None or enemy is None:
            print("Failed to find agent or enemy in observation.")
            return torch.zeros(FEATURE_SIZE, dtype=torch.float32)
        
        # Extract the agent's position and life
        agent_x, agent_z, agent_x_motion, agent_z_motion, agent_yaw, agent_life = (
            agent["x"],
            agent["z"],
            agent["motionX"],
            agent["motionZ"],
            agent["yaw"],
            agent["life"]
        )

        # Bound yaw between -180 and 180
        agent_yaw = bound_angle(agent_yaw)

        # Calculate the agent's overall speed
        agent_speed = math.sqrt(agent_x_motion**2 + agent_z_motion**2)

        # Calculate the change in yaw from the previous step
        agent_yaw_delta = agent_yaw - previous_agent_yaw

        # Bound between -180 and 180
        agent_yaw_delta = bound_angle(agent_yaw_delta)

        # Extract the enemy's position and life
        enemy_x, enemy_z, enemy_life = enemy["x"], enemy["z"], enemy["life"]

        # Calculate the distance & angle to the enemy
        dx, dz = enemy_x - agent_x, enemy_z - agent_z

        distance_to_enemy = math.sqrt(dx**2 + dz**2)

        yaw_to_enemy = -180 * math.atan2(dx, dz) / math.pi
        
        enemy_yaw_delta = agent_yaw - yaw_to_enemy
        
        # Bound between -180 and 180
        enemy_yaw_delta = bound_angle(enemy_yaw_delta)

        # Build the state vector
        state = torch.tensor(
            [agent_speed, agent_yaw_delta, agent_yaw, agent_life, distance_to_enemy, enemy_yaw_delta, enemy_life],
            dtype=torch.float32,
        )

        return state
    except:
        print("Failed to extract state from observation.")
        return torch.zeros(FEATURE_SIZE, dtype=torch.float32)
    
    
def calculate_speed(end_pos, start_pos):
    """Calculate the speed of the agent in the x and z directions."""
    x_speed = (end_pos[0] - start_pos[0])
    z_speed = (end_pos[2] - start_pos[2])
    speed = math.sqrt(x_speed**2 + z_speed**2)
    return speed