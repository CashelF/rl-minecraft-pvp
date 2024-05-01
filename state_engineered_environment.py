import gym
from gym import spaces
import torch
import math
import numpy as np

FEATURE_SIZE = 7

class StateEngineeredEnvironment(gym.Wrapper):
    def __init__(self, env):
        super(StateEngineeredEnvironment, self).__init__(env)
        self.previous_agent_yaw = 0.0
        self.current_damage_dealt = 0.0
        low_limits = np.array([
            0.0,    # agent_speed min
            -180.0, # agent_yaw_delta min
            -180.0, # agent_yaw min
            0.0,    # agent_life min
            0.0,    # distance to enemy min
            -180.0, # enemy_yaw_delta min
            0.0     # enemy_life min
        ], dtype=np.float32)

        high_limits = np.array([
            np.inf, # agent_speed max (set to infinity)
            180.0,  # agent_yaw_delta max
            180.0,  # agent_yaw max
            20.0,   # agent_life max
            10.0,   # distance to enemy max
            180.0,  # enemy_yaw_delta max
            20.0    # enemy_life max
        ], dtype=np.float32)

        # Set the observation space with the defined limits
        self.observation_space = spaces.Box(low=low_limits, high=high_limits, dtype=np.float32)

    def reset(self):
        info = self.env.reset()
        self.previous_agent_yaw = 0.0  # Reset the previous yaw with each new episode
        return self.encode_state(info)

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        new_state, reward = self.encode_state(info, reward)
        return frame, reward, done, info
    
    def encode_state(self, info: dict, reward: float = 0.0):
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
            agent_yaw = agent_yaw + 360 if agent_yaw < -180 else agent_yaw - 360 if agent_yaw > 180 else agent_yaw

            # Calculate the agent's overall speed
            agent_speed = math.sqrt(agent_x_motion**2 + agent_z_motion**2)

            # Calculate the change in yaw from the previous step
            agent_yaw_delta = agent_yaw - self.previous_agent_yaw

            # Bound between -180 and 180
            agent_yaw_delta = agent_yaw_delta + 360 if agent_yaw_delta < -180 else agent_yaw_delta - 360 if agent_yaw_delta > 180 else agent_yaw_delta

            # Extract the enemy's position and life
            enemy_x, enemy_z, enemy_life = enemy["x"], enemy["z"], enemy["life"]

            # Calculate the distance & angle to the enemy
            dx, dz = enemy_x - agent_x, enemy_z - agent_z

            distance_to_enemy = math.sqrt(dx**2 + dz**2)

            yaw_to_enemy = -180 * math.atan2(dx, dz) / math.pi
            
            enemy_yaw_delta = agent_yaw - yaw_to_enemy
            
            # Bound between -180 and 180
            enemy_yaw_delta = enemy_yaw_delta + 360 if enemy_yaw_delta < -  180 else enemy_yaw_delta - 360 if enemy_yaw_delta > 180 else enemy_yaw_delta
            
            try:
                if info["observation"]["DamageDealt"] > self.current_damage_dealt:
                    reward += (info["observation"]["DamageDealt"] - self.current_damage_dealt) / 10
                    self.current_damage_dealt = info["observation"]["DamageDealt"]
                
                if info["observation"]["PlayersKilled"] > 0:
                    print(f"{info['observation']['Name']} killed the enemy!")
                    super(StateEngineeredEnvironment, self).agent_host.sendCommand("quit")
                    reward += 100
                    done = True   

                # Update the current yaw measurement
                self.previous_agent_yaw = agent_yaw
            except:
                print("Too bad!")
            
            # Build the state vector
            state = torch.tensor(
                [agent_speed, agent_yaw_delta, agent_yaw, agent_life, distance_to_enemy, enemy_yaw_delta, enemy_life],
                dtype=torch.float32,
            )

            return state, reward
        except:
            print("Failed to extract state from observation.")
            return torch.zeros(FEATURE_SIZE, dtype=torch.float32)
