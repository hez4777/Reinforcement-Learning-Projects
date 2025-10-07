from MDP import build_infinite_horizon_maze_MDP, build_finite_horizon_maze_MDP
from IH import InfiniteHorizon
from DP import DynamicProgramming
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
import numpy as np
import argparse
import pdb


class Visualize:
    def __init__(self, DP):
        self.DP = DP

        self.arrows = ["\u2191", "\u2193", "\u2190", "\u2192"]
        self.iteration = 0

    def visualize_dp(self, policy, value_function):
        horizon = self.DP.horizon
        self.visualize_V_and_Pi(value_function, policy)

    def visualize_value_iteration(self, initialQ):
        self.Q = initialQ
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(
            f"Q-function values for iteration {self.iteration} of value iteration"
        )

        Q_up = self.Q[:-1, 0].reshape(4, 4)
        Q_down = self.Q[:-1, 1].reshape(4, 4)
        Q_left = self.Q[:-1, 2].reshape(4, 4)
        Q_right = self.Q[:-1, 3].reshape(4, 4)

        a_img = {}
        a_img[0] = ax[0, 0].matshow(
            Q_up, cmap="seismic", vmin=np.min(Q_up) - 40, vmax=np.max(Q_up)
        )
        ax[0, 0].set_title("Up-action (a=0)")
        a_img[1] = ax[0, 1].matshow(
            Q_down, cmap="seismic", vmin=np.min(Q_down) - 40, vmax=np.max(Q_down)
        )
        ax[0, 1].set_title("Down-action (a=1)")
        a_img[2] = ax[1, 0].matshow(
            Q_left, cmap="seismic", vmin=np.min(Q_left) - 40, vmax=np.max(Q_left)
        )
        ax[1, 0].set_title("Left-action (a=2)")
        a_img[3] = ax[1, 1].matshow(
            Q_right, cmap="seismic", vmin=np.min(Q_right) - 40, vmax=np.max(Q_right)
        )
        ax[1, 1].set_title("Right-action (a=3)")

        text_dictionary = {0: {}, 1: {}, 2: {}, 3: {}}
        for i, j in np.ndindex(Q_up.shape):
            text_dictionary[0][(i, j)] = ax[0, 0].text(
                j, i, "{:0.1f}".format(Q_up[i][j]), ha="center", va="center"
            )
            text_dictionary[1][(i, j)] = ax[0, 1].text(
                j, i, "{:0.1f}".format(Q_down[i][j]), ha="center", va="center"
            )
            text_dictionary[2][(i, j)] = ax[1, 0].text(
                j, i, "{:0.1f}".format(Q_left[i][j]), ha="center", va="center"
            )
            text_dictionary[3][(i, j)] = ax[1, 1].text(
                j, i, "{:0.1f}".format(Q_right[i][j]), ha="center", va="center"
            )

        fig.canvas.mpl_connect(
            "key_press_event",
            lambda event: self.VI_on_keyboard(event, fig, a_img, text_dictionary),
        )

        plt.show()

    def VI_on_keyboard(self, event, fig, a_grid, text_dic):
        if event.key == "right":
            self.iteration += 1
            self.Q = self.DP.valueIterationStep(self.Q)
            Q_up = self.Q[:-1, 0].reshape(4, 4)
            Q_down = self.Q[:-1, 1].reshape(4, 4)
            Q_left = self.Q[:-1, 2].reshape(4, 4)
            Q_right = self.Q[:-1, 3].reshape(4, 4)
            a_grid[0].set_array(Q_up)
            a_grid[1].set_array(Q_down)
            a_grid[2].set_array(Q_left)
            a_grid[3].set_array(Q_right)
            for i, j in np.ndindex(Q_up.shape):
                text_dic[0][(i, j)].set_text("{:0.1f}".format(Q_up[i][j]))
                text_dic[1][(i, j)].set_text("{:0.1f}".format(Q_down[i][j]))
                text_dic[2][(i, j)].set_text("{:0.1f}".format(Q_left[i][j]))
                text_dic[3][(i, j)].set_text("{:0.1f}".format(Q_right[i][j]))

            a_grid[0].set_clim(vmin=np.min(Q_up) - 40, vmax=np.max(Q_up))
            a_grid[1].set_clim(vmin=np.min(Q_down) - 40, vmax=np.max(Q_down))
            a_grid[2].set_clim(vmin=np.min(Q_left) - 40, vmax=np.max(Q_left))
            a_grid[3].set_clim(vmin=np.min(Q_right) - 40, vmax=np.max(Q_right))

        elif event.key == "enter":
            pi = self.DP.extractMaxPifromQ(self.Q)
            V_pi = self.DP.computeVfromQ(self.Q, pi)
            self.visualize_V_and_Pi(V_pi, pi)
        """
        elif event.key=='enter':
          
            self.policy = self.DP.extractPolicyfromV(self.V)
            data = np.reshape(self.policy[:-1], (4,4))
            for (i, j), z in np.ndenumerate(data):
                b_dic[(i,j)].set_text(self.arrows[z])
            self.iteration += 1
        """
        fig.suptitle(
            f"Q-function values for iteration {self.iteration} of value iteration"
        )

        fig.canvas.flush_events()
        fig.canvas.draw()

    def visualize_policy_iteration(self, initial_pi, exact):
        self.pi = initial_pi
        fig, ax = plt.subplots(2)
        self.V = (
            self.DP.exactPolicyEvaluation(initial_pi)
            if exact
            else self.DP.approxPolicyEvaluation(initial_pi)[0]
        )
        V_data = np.reshape(self.V[:-1], (4, 4))
        policy_data = np.reshape(self.pi[:-1], (4, 4))
        a_img = ax[0].matshow(
            V_data, cmap="seismic", vmin=np.min(self.V) - 40, vmax=np.max(self.V)
        )
        b_img = ax[1].matshow(
            np.zeros((4, 4)), cmap=ListedColormap(["w", "w", "w", "w"])
        )

        a_text_dictionary = {}
        b_text_dictionary = {}
        for (i, j), z in np.ndenumerate(V_data):
            a_text_dictionary[(i, j)] = ax[0].text(
                j, i, "{:0.1f}".format(z), ha="center", va="center"
            )

        for (i, j), z in np.ndenumerate(policy_data):
            b_text_dictionary[(i, j)] = ax[1].text(
                j, i, self.arrows[z], ha="center", va="center"
            )
        fig.canvas.mpl_connect(
            "key_press_event",
            lambda event: self.PI_on_keyboard(
                event, exact, fig, a_img, b_img, a_text_dictionary, b_text_dictionary
            ),
        )

        fig.suptitle(
            f"Iteration {self.iteration} of {'exact' if exact else 'approximate'} policy iteration"
        )

        plt.show()

    def PI_on_keyboard(self, event, exact, fig, a_grid, b_grid, a_dic, b_dic):
        if event.key == "right":
            self.pi = self.DP.policyIterationStep(self.pi, exact)
            self.V = (
                self.DP.exactPolicyEvaluation(self.pi)
                if exact
                else self.DP.approxPolicyEvaluation(self.pi)[0]
            )

            V_data = np.reshape(self.V[:-1], (4, 4))
            a_grid.set_array(V_data)
            for (i, j), z in np.ndenumerate(V_data):
                a_dic[(i, j)].set_text("{:0.1f}".format(z))

            a_grid.set_clim(vmin=np.min(self.V) - 40)
            a_grid.set_clim(vmax=np.max(self.V))

            policy_data = np.reshape(self.pi[:-1], (4, 4))
            b_grid.set_array(policy_data)
            for (i, j), z in np.ndenumerate(policy_data):
                b_dic[(i, j)].set_text(self.arrows[z])
            self.iteration += 1
            fig.suptitle(
                f"Iteration {self.iteration} of {'exact' if exact else 'approximate'} policy iteration"
            )
            fig.canvas.flush_events()
            fig.canvas.draw()

    def visualize_V_and_Pi(self, V, pi):
        fig, ax = plt.subplots(2)
        if len(V.shape) == 2:
            V_data = np.reshape(V[0][:-1], (4, 4))
            pi_data = np.reshape(pi[0][:-1].astype(int), (4, 4))
        else:
            V_data = np.reshape(V[:-1], (4, 4))
            pi_data = np.reshape(pi[:-1].astype(int), (4, 4))
        a_img = ax[0].matshow(
            V_data, cmap="seismic", vmin=np.min(V) - 40, vmax=np.max(V)
        )
        b_img = ax[1].matshow(np.zeros((4, 4)), vmin=-1, vmax=1)

        a_text_dictionary = {}
        b_text_dictionary = {}
        for i, j in np.ndindex(V_data.shape):
            a_text_dictionary[(i, j)] = ax[0].text(
                j, i, "{:0.1f}".format(V_data[i, j]), ha="center", va="center"
            )
            b_text_dictionary[(i, j)] = ax[1].text(
                j, i, self.arrows[pi_data[i, j]], ha="center", va="center"
            )

        fig.canvas.mpl_connect(
            "key_press_event",
            lambda event: self.DP_on_keyboard(
                event, V, pi, fig, a_img, b_img, a_text_dictionary, b_text_dictionary
            ),
        )

        fig.suptitle(f"Value Function and Policy on t={self.iteration}")

        plt.show()

    def DP_on_keyboard(self, event, V, pi, fig, a_grid, b_grid, a_dic, b_dic):
        if event.key == "right":
            self.iteration += 1
            if self.iteration >= len(pi)-1:
                index = len(pi)-1
            else:
                index = self.iteration
            self.pi = pi[index]
            self.V = V[index]

            V_data = np.reshape(self.V[:-1], (4, 4))
            a_grid.set_array(V_data)
            for (i, j), z in np.ndenumerate(V_data):
                a_dic[(i, j)].set_text("{:0.1f}".format(z))

            a_grid.set_clim(vmin=np.min(self.V) - 40)
            a_grid.set_clim(vmax=np.max(self.V))

            policy_data = np.reshape(self.pi[:-1].astype(int), (4, 4))
            b_grid.set_array(policy_data)
            for (i, j), z in np.ndenumerate(policy_data):
                b_dic[(i, j)].set_text(self.arrows[z])

            fig.suptitle(f"Value Function and Policy on t={index}")

            fig.canvas.flush_events()
            fig.canvas.draw()


if __name__ == "__main__":
    # read system arguments argparse
    parser = argparse.ArgumentParser(
        prog="GridWorldVisualizer",
        description="Visualizer for gridworld MDPs",
        epilog="Example: python3 visualize.py --alg VI",
    )
    parser.add_argument(
        "--alg",
        nargs=1,
        help="Algorithm to visualize. Options: VI, PI_exact, PI_approx, DP",
    )
    parser.add_argument(
        "--gamma", "-G", nargs=1, help="Discount factor: 0 <= gamma < 1"
    )
    parser.add_argument("--horizon", "-H", nargs=1, help="Horizon, H \ge 1")

    args = parser.parse_args()

    # asserts
    assert args.alg[0] in ["VI", "PI_exact", "PI_approx", "DP"], "Invalid algorithm"
    if args.gamma is not None and args.horizon is not None:
        print("Warning: cannot specify both gamma and horizon. Using horizon.")
    if args.horizon is not None:
        assert int(args.horizon[0]) >= 1, "Invalid horizon"
    if args.gamma is not None:
        assert 0 <= float(args.gamma[0]) < 1, "Invalid gamma"

    if args.alg[0] == "DP":
        assert args.horizon is not None, "Must specify horizon"

    if args.alg[0] == "DP":
        mdp = build_finite_horizon_maze_MDP(int(args.horizon[0]))
        dp = DynamicProgramming(mdp)
        policy, value_function = dp.dynamicProgramming()
        v = Visualize(dp)
        v.visualize_dp(policy, value_function)
    else:
        mdp = build_infinite_horizon_maze_MDP(float(args.gamma[0]))
        dp = InfiniteHorizon(mdp)
        v = Visualize(dp)
        if args.alg[0] == "VI":
            Q = np.zeros((dp.nStates, dp.nActions))
            v.visualize_value_iteration(Q)
        elif args.alg[0] == "PI_exact":
            pi = np.zeros(dp.nStates, dtype="int")
            v.visualize_policy_iteration(pi, True)
        else:
            pi = np.zeros(dp.nStates, dtype="int")
            v.visualize_policy_iteration(pi, False)
