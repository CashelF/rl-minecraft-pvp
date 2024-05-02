import os
import random
import threading
from collections import deque

import marlo
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils import FEATURE_SIZE, NUM_ACTIONS, build_model, encode_state


def play_episodes(
    env,
    model: nn.Module,
    episodes: int = 1000,
    epsilon: float = 0,
):

    # Set the model to evaluation mode
    model.eval()

    
    for episode in range(episodes):

        # Reset the environment
        frame = env.reset()

        # First random step
        action = env.action_space.sample()

        # Step the environment
        frame, reward, done, info = env.step(action)

        state = encode_state(info)
        
        current_damage_dealt = 0
        current_damage_taken = 0

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
                    # Get the model's prediction
                    logits = model(state)

                    # Sample from the distribution
                    action_probs = torch.softmax(logits, dim=0)

                    # Sample an action from the distribution
                    # action = torch.multinomial(action_probs, num_samples=1).item()
                    action = torch.argmax(action_probs).item()

            # Step the environment
            frame, reward, done, info = env.step(action)

            # Preprocess the next state
            next_state = encode_state(info)

            # Update the state
            state = next_state
@marlo.threaded
def start_agent(join_token, model: nn.Module):
    # Initialize the environment
    env = marlo.init(join_token)

    # Play the episodes
    play_episodes(env, model)

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

    model.load_state_dict(torch.load("models/model.pt"))
    threads = []

    threads.append(start_agent(join_tokens[0], model)[0])
    threads.append(start_agent(join_tokens[1], model)[0])

    # Wait for training to finish
    for thread in threads:
        thread.join()

    print("Training Complete")