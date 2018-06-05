import numpy as np


class GuessingSumEnv:
    def __init__(self, num_agents=5):
        self.num_agents = num_agents
        self.sum = 0
        self.scale = 10.0
        self.sum_scale = self.num_agents * self.scale

    def step(self, actions):
        if actions.shape != (self.num_agents, 1):
            raise Exception('got input shape ', actions.shape, ' instead of ', (self.num_agents, 1))

        observations = None
        rewards = -np.abs(actions - self.sum) # [-Inf ; 0]

        normalized_rewards = (np.maximum(rewards, -self.sum_scale) + self.sum_scale) / self.sum_scale # [0 ; 1]

        done = True
        info = None

        return observations, normalized_rewards, done, info

    def reset(self):
        observations = np.clip(np.random.normal(size=(self.num_agents, 1)), -self.scale, self.scale)
        self.sum = np.sum(observations)
        return observations

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed=None):
        np.random.seed(seed)
        return


if __name__ == '__main__':
    env = GuessingSumEnv()
    env.seed(0)

    print('obs:', env.reset())
    actions = np.random.normal(size=(env.num_agents, 1))
    print('actions:', actions)
    print('rewards:', env.step(actions))
