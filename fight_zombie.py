import json

import marlo
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from dqn import DQN, train_dqn


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


def resize_and_normalize(img, size=(150, 100)):
    # Convert numpy array to PIL Image for resizing
    img = TF.to_pil_image(img)
    img = TF.resize(img, size)
    # Convert back to tensor and normalize
    img = TF.to_tensor(img)
    img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization values
    return img

if __name__ == "__main__":
    env = init_env()
    
    dqn_model = DQN(env.action_space.n)
    # Image Preprocessing
    transform = transforms.Lambda(lambda img: resize_and_normalize(img))
    
    train_dqn(env, dqn_model, transform)
    # example(env)
    