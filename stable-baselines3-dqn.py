import gym
import marlo
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from environment import ZombieEnv



def initialize_environment(mission_file: str = "mission.xml"):
    # Define the client pool
    client_pool = [('127.0.0.1', 10000)]
    # Create the environment. 'suppress_info' hides unnecessary outputs.
    join_tokens = marlo.make(mission_file, params={"client_pool": client_pool, "suppress_info": False})
    # Initialize the environment (Assuming only one agent)
    env = marlo.init(join_tokens[0])
    env = ZombieEnv(env)
    return env

if __name__ == "__main__":
    # Initialize the environment
    env = initialize_environment()

    # Stable Baselines3 requires the environment to be wrapped correctly with Gym
    # If Marlo env is not fully compliant with Gym, you might need to create a wrapper
    # Assuming env is now a compliant Gym environment

    # Define the policy architecture (can use a custom policy or predefined ones)
    # Here we use a simple Multi-Layer Perceptron with two layers of 256 neurons each
    model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000, learning_rate=0.0005,
                batch_size=64, gamma=0.99, exploration_fraction=0.1, exploration_final_eps=0.02, 
                target_update_interval=1000, train_freq=4, gradient_steps=1, tensorboard_log="./dqn_minecraft_tensorboard/")

    # Train the model
    model.learn(total_timesteps=25000)

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Save the model
    model.save("models/dqn_minecraft_model")

    # Close the environment when done
    env.close()
