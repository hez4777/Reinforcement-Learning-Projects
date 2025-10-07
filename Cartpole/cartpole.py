import gym
import env
import numpy as np
from pa2.lqr import LQRController
from pa2.ddp import DDPController
from gym import wrappers
import argparse
from time import sleep
from tqdm import tqdm
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CartpoleVisualizer",
        description="Visualizer for trained cartpole controllers",
        epilog="Example: python3 cartpole.py --env LQR",
    )
    parser.add_argument(
        "--env",
        nargs=1,
        help="Environment to visualize. Options: LQR, DDP",
    )
    parser.add_argument(
        "--init_s",
        nargs=1,
        help="Initial distribution for DDP tests. Options: close, mid, far",
    )
    
    args = parser.parse_args()
    if args.env is not None:
        assert args.env[0] in ["LQR", "DDP"], "Invalid environment"
        if args.env[0]=="DDP":
            if args.init_s is not None:
                assert args.init_s[0] in ["close", "mid", "far"], "Invalid initial state"
                if args.init_s[0]=="far":
                    init_state = [0.0, 0.0, 0.5 * np.pi, 0.0] 
                elif args.init_s[0]=="mid":
                    init_state = [0.0, 0.0, 0, 0.0]
                elif args.init_s[0]=="close":
                    init_state = [0.0, 0.0, -0.5 * np.pi, 0.0]
            else:
                init_state = [0.0, 0.0, 0.5 * np.pi, 0.0]
            
            env = gym.make('CartPoleILQREnv-v0').env
            obs = env.reset(init_state)
            ddp = DDPController(env)
            
            actions = ddp.train(obs)
            while True:
                env.reset(init_state)
                for action in actions:
                    env.render()
                    env.step(action)
                env.close()

        else:
            env = gym.make("CartPoleLQREnv-v0")
            s_star = np.array([0, 0, 0, 0], dtype=np.double)
            a_star = np.array([0], dtype=np.double)
            T = 500
            controller = LQRController(env)
            policies = controller.lqr(s_star,a_star,T)

            # For testing, we use a noisy environment which adds small Gaussian noise to
            # state transition. Your controller only need to consider the env without noise.
            env = gym.make("NoisyCartPoleLQREnv-v0")

            video_path = "./gym-results"
            init_states = [np.array([0.0, 0.0, 0.0, 0.0]),
                   np.array([0.0, 0.0, 0.2, 0.0]),
                   np.array([0.0, 0.0, 0.4, 0.0]),
                   np.array([0.0, 0.0, 0.6, 0.0]),
                   np.array([0.0, 0.0, 0.8, 0.0]),
                   np.array([0.0, 0.0, 1.0, 0.0]),
                   np.array([0.0, 0.0, 1.2, 0.0]),
                   np.array([0.0, 0.0, 1.4, 0.0])]
            s_star = np.array([0, 0, 0, 0], dtype=np.double)
            a_star = np.array([0], dtype=np.double)
            T = 500

            for init_state in init_states:
                total_cost = 0
                observation = env.reset(init_state)

                for t in range(T):
                    env.render()
                    (K,k) = policies[t]
                    action = (K @ observation + k)
                    observation, cost, done, info = env.step(action)
                    total_cost += cost
                    if done: # When the state is out of the range, the cost is set to inf and done is set to True
                        break
                print("cost = ", total_cost)
    else:
        print("Please specify an environment to visualize with --env {LQR, DDP}")
        exit(1)



 