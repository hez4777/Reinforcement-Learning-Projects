import torch
import numpy as np
import gym
import argparse
import sys


def test_agent(agent_path, env):
    lengths = []
    rewards = []
    episode_length = 10 if 'cartpole-v0' in agent_path else 100
    for _ in range(episode_length):
        done = False
        ob = env.reset()
        agent = torch.load(agent_path)
        length = 0
        reward = 0

        while not done:
            if 'google.cloud' not in sys.modules:  # env.render() will not work in colab
                env.render()
            qs = agent(torch.from_numpy(ob).float())
            a = qs.argmax().numpy()

            next_ob, r, done, _ = env.step(a)
            ob = next_ob
            length += 1
            reward += r

        env.close()
        lengths.append(length)
        rewards.append(reward)

    print(f'average episode length: {np.mean(lengths)}')
    print(f'average reward incurred: {np.mean(rewards)}')


def get_args():
    parser = argparse.ArgumentParser(description='test-function')
    parser.add_argument('--env', default='CartPole-v0', help='name of environment')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    env = gym.make(args.env)
    env.seed(0)
    agent_path = f'./trained_agent_{args.env.lower()}.pt'
    test_agent(agent_path, env)
