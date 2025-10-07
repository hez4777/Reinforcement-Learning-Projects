from sklearn.kernel_approximation import RBFSampler
import numpy as np
import argparse

rbf_feature = RBFSampler(gamma=1, random_state=12345)


def extract_features(state, num_actions):
    """ This function computes the RFF features for a state for all the discrete actions

    :param state: column vector of the state we want to compute phi(s,a) of (shape |S|x1)
    :param num_actions: number of discrete actions you want to compute the RFF features for
    :return: phi(s,a) for all the actions (shape 100x|num_actions|)
    """
    s = state.reshape(1, -1)
    s = np.repeat(s, num_actions, 0)
    a = np.arange(0, num_actions).reshape(-1, 1)
    sa = np.concatenate([s,a], -1)
    feats = rbf_feature.fit_transform(sa)
    feats = feats.T
    return feats


def compute_softmax(logits, axis):
    """ computes the softmax of the logits

    :param logits: the vector to compute the softmax over
    :param axis: the axis we are summing over
    :return: the softmax of the vector

    Hint: to make the softmax more stable, subtract the max from the vector before applying softmax
    """

    # TODO
    exp = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    softmax = exp / np.sum(exp, axis=axis, keepdims=True)
    return softmax
    pass


def compute_action_distribution(theta, phis):
    """ compute probability distribution over actions

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :return: probability distribution over actions that is computed via softmax (shape 1 x |A|)
    """

    # TODO
    logits = np.dot(theta.T, phis)
    prob = compute_softmax(logits, axis=1)
    return prob
    pass


def compute_log_softmax_grad(theta, phis, action_idx):
    """ computes the log softmax gradient for the action with index action_idx

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :param action_idx: The index of the action you want to compute the gradient of theta with respect to
    :return: log softmax gradient (shape d x 1)
    """

    # TODO
    prob = compute_action_distribution(theta, phis)
    grad = phis[:, action_idx] - np.sum(phis * prob, axis=1)
    return grad[:, np.newaxis]
    pass


def compute_fisher_matrix(grads, lamb=1e-3):
    """ computes the fisher information matrix using the sampled trajectories gradients

    :param grads: list of list of gradients, where each sublist represents a trajectory (each gradient has shape d x 1)
    :param lamb: lambda value used for regularization 

    :return: fisher information matrix (shape d x d)

    Note: don't forget to take into account that trajectories might have different lengths
    """

    # TODO
    d = grads[0][0].shape[0]
    F = np.zeros((d, d))
    total_samples = 0

    for grad_trajectory in grads:
        H_i = len(grad_trajectory)
        for grad in grad_trajectory:
            F += np.dot(grad, grad.T) / H_i
        total_samples += 1

    F /= total_samples 
    F += lamb * np.eye(d) 
    return F
    #pass


def compute_value_gradient(grads, rewards):
    """ computes the value function gradient with respect to the sampled gradients and rewards

    :param grads: ist of list of gradients, where each sublist represents a trajectory
    :param rewards: list of list of rewards, where each sublist represents a trajectory
    :return: value function gradient with respect to theta (shape d x 1)
    """

    # TODO
    num_trajectories = len(grads)
    d = grads[0][0].shape[0]
    
    total_rewards = [np.sum(reward_traj) for reward_traj in rewards]
    baseline = np.mean(total_rewards)
    v_grad = np.zeros((d, 1))

    for grad_traj, reward_traj in zip(grads, rewards):
        traj_length = len(grad_traj)
        
        discounted_rewards = np.zeros_like(reward_traj)
        cumulative_reward = 0
        for t in reversed(range(traj_length)):
            cumulative_reward = reward_traj[t] + cumulative_reward
            discounted_rewards[t] = cumulative_reward
        
        normalized_rewards = discounted_rewards - baseline
        traj_grad = np.zeros((d, 1))
        for t in range(traj_length):
            traj_grad += grad_traj[t] * normalized_rewards[t]
        
        v_grad += traj_grad / traj_length
    
    v_grad /= num_trajectories
    
    return v_grad
    #pass



def compute_eta(delta, fisher, v_grad):
    """ computes the learning rate for gradient descent

    :param delta: trust region size
    :param fisher: fisher information matrix (shape d x d)
    :param v_grad: value function gradient with respect to theta (shape d x 1)
    :return: the maximum learning rate that respects the trust region size delta
    """

    # TODO
    quadratic_term = np.dot(v_grad.T, np.linalg.solve(fisher, v_grad))
    eta = np.sqrt(delta / (quadratic_term + 1e-6))
    return eta
    pass



def get_args():
    parser = argparse.ArgumentParser(description='Imitation learning')

    # general + env args
    parser.add_argument('--data_dir', default='./data', help='dataset directory')
    parser.add_argument('--env', default='CartPole-v0', help='environment')
    
    # learning args
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_dataset_samples', type=int, default=10000, help='number of samples to start dataset off with')
    
    # DAGGER args
    parser.add_argument('--dagger', action='store_true', help='flag to run DAGGER')
    parser.add_argument('--expert_save_path', default='./learned_policies/NPG/expert_theta.npy')
    parser.add_argument('--num_rollout_steps', type=int, help='number of steps to roll out with the policy')
    parser.add_argument('--dagger_epochs', type=int, help='number of steps to run dagger')
    parser.add_argument('--dagger_supervision_steps', type=int, help='number of epochs for supervised learning step within dagger')
    
    # model saving args
    parser.add_argument('--policy_save_dir', default='./learned_policies', help='policy saving directory')
    parser.add_argument('--state_to_remove', default=None, type=int, help='index of the state to remove')
    
    return parser.parse_args()