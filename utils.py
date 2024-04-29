import datetime
import math
import pickle
import random
import time

import torch
import torch.nn as nn

FEATURE_SIZE = 7
NUM_ACTIONS = 7

def get_new_filename():
    """Generate a unique filename using the current timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"trajectory_data_{timestamp}.pkl"

def save_trajectory(filename, trajectory):
    """Save the complete trajectory list to a binary file using pickle."""
    with open(filename, 'wb') as file:  # 'wb' for writing in binary
        pickle.dump(trajectory, file)

def load_trajectory_data(filename):
    """Load the entire list of trajectory data from a pickle file."""
    with open(filename, 'rb') as file:
        trajectory = pickle.load(file)
    return trajectory


def build_model(num_inputs: int, num_actions: int, hidden_size: int = 512, dropout_prob: float = 0.4) -> nn.Module:
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
        nn.Linear(hidden_size, num_actions),
    )

    return model

def randomly_move_agent(env, num_steps_range=(1, 10), num_turns_range=(0, 4)):
    """Move the agent randomly at the start of the episode to simulate changing spawn points.
    
    Args:
        env: The environment instance for the agent.
        num_steps_range (tuple): A tuple specifying the min and max number of steps the agent should move forward.
        num_turns_range (tuple): A tuple specifying the min and max number of 90-degree turns the agent should make.
    """
    num_steps = random.randint(*num_steps_range)
    num_turns = random.randint(*num_turns_range)
    turn_direction = random.choice(["turn 1", "turn -1"])  # 'turn 1' for right, 'turn -1' for left

    # Execute turn commands
    for _ in range(num_turns):
        env.agent_host.sendCommand(turn_direction)
        time.sleep(0.5)  # Sleep to ensure the command is executed before the next one

    env.agent_host.sendCommand("turn 0")

    # Move forward
    for _ in range(num_steps):
        env.agent_host.sendCommand("move 1")
        time.sleep(0.5)

    env.agent_host.sendCommand("move 0")
    

def preprocess_observation(info):
    """Extract the hand-crafted state vector from an observation"""
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
                
                yaw_delta = yaw - yaw_to_enemy
                
                # Bound yaw_delta between -180 and 180
                yaw_delta = yaw_delta + 360 if yaw_delta < -180 else yaw_delta - 360 if yaw_delta > 180 else yaw_delta

                # Build the state vector
                state = torch.tensor(
                    [x, z, yaw, life, distance_to_enemy, yaw_delta, enemy_life],
                    dtype=torch.float32,
                )

                return state

        # No enemy found
        return torch.zeros(FEATURE_SIZE, dtype=torch.float32)
    except:
        # Observation not found
        return torch.zeros(FEATURE_SIZE, dtype=torch.float32)