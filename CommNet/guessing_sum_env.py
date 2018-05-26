import numpy as np


class GuessingSumEnv:
    def __init__(self, num_agents=5):
        self.num_agents = num_agents
        self.sum = 0

    def step(self, actions):
        if actions.shape != (self.num_agents, 1):
            raise Exception('got input shape ', actions.shape, ' instead of ', (self.num_agents, 1))

        observations = None
        rewards = -np.abs(actions - self.sum) / self.num_agents

        done = True
        info = None

        return observations, rewards, done, info

    def reset(self):
        observations = np.random.uniform(low=-1.0, high=1.0, size=(self.num_agents, 1)) / self.num_agents
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
