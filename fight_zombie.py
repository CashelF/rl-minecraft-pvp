import json

import marlo

from dqn import create_dqn_model, train_dqn


def init_env():
    # Load the mission XML from a string or file
    mission_xml = "./fight_zombie_env.xml"  # Assume XML content is saved here

    client_pool = [('127.0.0.1', 10000)]
    join_tokens = marlo.make(mission_xml, params={"client_pool": client_pool, 'videoResolution': [800, 600]})

    join_token = join_tokens[0]
    env = marlo.init(join_token)
    
    return env

def example(env):
    for episode in range(5):
        print("Starting episode:", episode)
        observation = env.reset()
        done = False
        while not done:
            # The agent takes random actions. Replace with your AI logic.
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print("Reward:", reward, "Done:", done)

            # Extract and use observations
            if "entities" in obs:
                entities_info = json.loads(obs["entities"])
                print("Entities observed:", entities_info)

        env.close()


if __name__ == "__main__":
    env = init_env()
    
    train_dqn(env)
    