from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import marlo
import torch
from stable_baselines3.common.logger import configure

def initialize_environments(client_pool, mission_file: str = "mission.xml"):
    join_tokens = marlo.make(mission_file, params={"client_pool": client_pool, "suppress_info": False})
    envs = [marlo.init(token) for token in join_tokens]
    return envs

@marlo.threaded
def start_agent(join_token, agent_id):
    # Initialize the environment
    env = marlo.init(join_token)
    tensorboard_path = f"./dqn_minecraft_tensorboard/agent_{agent_id}"
    logger = configure(tensorboard_path, ["stdout", "csv", "tensorboard"])
    model = DQN("MlpPolicy", env, verbose=1, buffer_size=10, learning_rate=0.0005,
                  batch_size=64, gamma=0.99, exploration_fraction=0.1, exploration_final_eps=0.02,
                  target_update_interval=1000, train_freq=4, gradient_steps=1, 
                  tensorboard_log=tensorboard_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    model.set_logger(logger)
    
    model.learn(total_timesteps=25000)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Agent {agent_id} Model - Mean reward: {mean_reward} +/- {std_reward}")
    model.save(f"models/dqn_minecraft_model_{agent_id}")
        
    env.close()

if __name__ == "__main__":
    client_pool = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
    join_tokens = marlo.make('mission.xml', params={"client_pool": client_pool, "suppress_info": False})

    threads = []

    threads.append(start_agent(join_tokens[0], 0))
    threads.append(start_agent(join_tokens[1], 1))

    # Wait for training to finish
    for thread in threads:
        thread.join()

    print("Training Complete")
