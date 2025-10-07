import numpy as np
from pa2.finite_difference_method import gradient, jacobian, hessian


class LocalController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset
                 the state to any state
        """
        self.env = env

    def c(self, s, a):
        """
        Cost function of the env.
        It sets the state of environment to `s` and then execute the action `a`, and
        then return the cost.
        Parameter:
            s (1D numpy array) with shape (4,)
            a (1D numpy array) with shape (1,)
        Returns:
            cost (double)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        observation, cost, done, info = env.step(a)
        return cost

    def f(self, s, a):
        """
        State transition function of the environment.
        Return the next state by executing action `a` at the state `s`
        Parameter:
            s (1D numpy array) with shape (4,)
            a (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        next_observation, cost, done, info = env.step(a)
        return next_observation

    def compute_local_expansions(self, s, a, f=None, c=None):
        """
        This function perform a first order taylar expansion function f and
        second order taylor expansion of cost function around (s, a). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            s (numpy array) with shape (4,)
            a (numpy array) with shape (1,)
        return
            A (2d numpy array): A numpy array with shape (n_s, n_s)
            B (2d numpy array): A numpy array with shape (n_s, n_a)
            Q (2d numpy array): A numpy array with shape (n_s, n_s). Q is PD
            R (2d numpy array): A numpy array with shape (n_a, n_a). R is PD.
            M (2d numpy array): A numpy array with shape (n_s, n_a)
            q (2d numpy array): A numpy array with shape (n_s, 1)
            r (2d numpy array): A numpy array with shape (n_a, 1)
        """
        if f is None:
            f = self.f
        if c is None:
            c = self.c

        # TODO
        
        A = jacobian(lambda s1: self.f(s1, a), s, delta=1e-5)
        B = jacobian(lambda a1: self.f(s, a1), a, delta=1e-5)

        q = gradient(lambda s1: self.c(s1, a), s, delta=1e-5)
        r = gradient(lambda a1: self.c(s, a1), a, delta=1e-5)

        Q = hessian(lambda s1: self.c(s1, a), s, delta=1e-5)
        R = hessian(lambda a1: self.c(s, a1), a, delta=1e-5)
        M = hessian(lambda sa: self.c(sa[:-1], sa[-1:]), np.concatenate((s, a)), delta=1e-5)[:-1, -1:]

        return A, B, Q, R, M, q, r
        pass