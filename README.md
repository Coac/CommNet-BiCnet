# CommNet-BiCnet
CommNet and BiCnet implementation in tensorflow

## Usage
```
python train_comm_net.py
```

## Guessing sum environment
It is a simple game described in the [BiCnet](https://arxiv.org/abs/1703.10069) paper for testing if the communication works. The environment implements the crucial methods of the core gym interface from OpenAI

Each agents receive a scalar sampled between `[−10, 10]` under a truncated gaussian. Each agent needs to output the sum of all inputs received among the agents. An agent gets a normalized reward between `[0, 1]` based on the absolute difference between the sum and its output.

## Results
### Training CommNet in the Guessing sum env with 2 agents
![2_agents_commnet_training_reward](docs/2_agents_commnet.png)
