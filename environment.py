import gym

from options import look_at_zombie

class ZombieEnv(gym.Env):
    def __init__(self, env):
        super(ZombieEnv, self).__init__()
        self.env = env
        # Set these based on your environment specifics
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = env.observation_space
        
        self.look_at_zombie_action = 9

    def step(self, action):
        done = False
        reward = 0
        if action == self.look_at_zombie_action:
            obs, reward, done, info = self.env.step(1) # TODO: is this a no-op aciton?
            look_at_zombie(info['observation'], self.env)
            reward = 0.01 # Reward for looking at the zombie (shaping reward)
        else:
            obs, reward, done, info = self.env.step(action)

        # Custom condition for ending the episode
        mobs_killed = info.get('observation', {}).get('MobsKilled', 0) if info['observation'] else 0
        if mobs_killed > 0:
            reward = 10
            done = True

        if reward != 0:
            print(f"Reward: {reward}")

        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()