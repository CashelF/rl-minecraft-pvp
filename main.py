import os
import random
import threading
from collections import deque

import marlo
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from train import train_random_sample
from utils import (
    FEATURE_SIZE,
    NUM_ACTIONS,
    bound_angle,
    build_model,
    encode_state,
    get_new_filename,
    randomly_move_agent,
    save_trajectory,
)

trajectory_file_name = get_new_filename()

def play_episodes(
    env,
    model: nn.Module,
    memory: deque,
    trajectories: list,
    episodes: int = 500,
    gamma: float = 0.9,
    initial_epsilon: float = 0.9,
    final_epsilon: float = 0.1,
    epsilon_decay: float = 0.99,
    log_dir: str = "logs/",
    model_dir: str = "models/",
    trajectory_dir: str = "trajectories/",
    primary_agent: bool = False,
):

    # Ensure the output directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(trajectory_dir, exist_ok=True)

    # Initialize epsilon
    epsilon = initial_epsilon

    if primary_agent:
        # Create a tensorboard writer
        writer = SummaryWriter(log_dir=log_dir)

        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Send the model to the device
        model.to(device)

    # Run the training loop
    for episode in range(episodes):

        if primary_agent:
            # Set the model to evaluation mode
            model.eval()

        # Reset the environment
        frame = env.reset()

        current_yaw = 0
        current_damage_dealt = 0

        episode_reward = 0
        episode_length = 0
        
        # Randomly move the agent to simulate random spawn points
        randomly_move_agent(env)

        # First random step
        action = env.action_space.sample()

        # Step the environment
        frame, reward, done, info = env.step(action)

        state = encode_state(info, 0)

        # Run the episode to completion
        done = False
        while not done:
            behavior_probabilty = 0

            if primary_agent: # Model Policy
                # Forward pass
                with torch.no_grad():
                    # Get the model's prediction
                    logits = model(state.to(device))

                    # Sample from the distribution
                    action_probs = torch.softmax(logits, dim=0)

                    # Bring back from the device
                    state = state.cpu()

                    # Sample an action from the distribution
                    action = env.action_space.sample() if random.random() < epsilon else torch.multinomial(action_probs, num_samples=1).item()

                # Calculate the behavior probability
                behavior_probabilty = epsilon * (1 / NUM_ACTIONS) + (1 - epsilon) * action_probs[action].item()
            else: # Hardcoded Policy
                speed, yaw_change, yaw, life, distance_to_enemy, yaw_delta, enemy_life = state

                if abs(yaw_delta) < 5: # If we are looking at the enemy
                    if distance_to_enemy < 3: #If they are close enough to hit, attack
                        action = 5 if action == 0 else 0
                    else: # If they are too far, move towards them
                        action = 1 if action not in [3, 4] else 0
                elif yaw_delta < 0: # Turn right
                    action = 3 if action != 1 else 0
                else: # Turn left
                    action = 4 if action != 1 else 0

                # Behavior probability is always 1 because the policy is deterministic
                behavior_probabilty = 1

            # Step the environment
            frame, reward, done, info = env.step(action)

            # Preprocess the next state
            next_state = encode_state(info, current_yaw)

            # This is very scuffed. There must be a better way to do this. Too bad!
            try:
                if info["observation"]["DamageDealt"] > current_damage_dealt:
                    reward += (info["observation"]["DamageDealt"] - current_damage_dealt) / 10
                    current_damage_dealt = info["observation"]["DamageDealt"]
                
                if info["observation"]["PlayersKilled"] > 0:
                    print(f"{info['observation']['Name']} killed the enemy!")
                    env.agent_host.sendCommand("quit")
                    reward += 100 - 0.1 * episode_length
                    done = True   

                # Update the current yaw measurement
                current_yaw = bound_angle(info["observation"]["Yaw"])
            except:
                print("Too bad!")

            # Update the episode reward and length
            episode_reward += reward
            episode_length += 1

            # Store the transition
            memory.append((state, action, reward, next_state, done, behavior_probabilty))
            
            # Add trajectory
            trajectories.append((state, action, reward, next_state, done, behavior_probabilty))   

            # Update the state
            state = next_state

        print(f"Episode {episode + 1} Reward({threading.current_thread()}): {episode_reward:.2f}")

        if primary_agent:
            # Train on trajectory data from the experience replay buffer
            loss = train_random_sample(model, memory, episode_length, gamma=gamma, device=device)

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

            print(f"Episode {episode + 1} Loss: {loss:.2f}")  


@marlo.threaded
def start_agent(join_token, memory, trajectories, model: nn.Module, can_train: bool = False):
    # Initialize the environment
    env = marlo.init(join_token)

    # Train for 2000 episodes
    play_episodes(env, memory, trajectories, model, episodes=4000, log_dir="logs/ImportanceSamplingTDUpdate", primary_agent=can_train)

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
    model = build_model(num_inputs=FEATURE_SIZE, num_actions=NUM_ACTIONS)

    # model.load_state_dict(torch.load("models/bot.pt"))

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