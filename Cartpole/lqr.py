import numpy as np
from pa2.local_controller import LocalController


class LQRController:
    def __init__(self, env):
        self.local_controller = LocalController(env)

    def compute_Q_params(self, A, B, Q, R, M, q, r, m, b, P, y, p):
        """
        Compute the Q function parameters for time step t.
        Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
            Parameters:
            A (2d numpy array): A numpy array with shape (n_s, n_s)
            B (2d numpy array): A numpy array with shape (n_s, n_a)
            Q (2d numpy array): A numpy array with shape (n_s, n_s). Q is PD
            R (2d numpy array): A numpy array with shape (n_a, n_a). R is PD.
            M (2d numpy array): A numpy array with shape (n_s, n_a)
            q (2d numpy array): A numpy array with shape (n_s, 1)
            r (2d numpy array): A numpy array with shape (n_a, 1)
            m (2d numpy array): A numpy array with shape (n_s, 1)
            b (1d numpy array): A numpy array with shape (1,)
            P (2d numpy array): A numpy array with shape (n_s, n_s). This is the quadratic term of the
                value function equation from time step t+1. P is PSD.
            y (2d numpy array): A numpy array with shape (n_s, 1).  This is the linear term
                of the value function equation from time step t+1
            p (1d numpy array): A numpy array with shape (1,).  This is the constant term of the
                value function equation from time step t+1
        Returns:
            C (2d numpy array): A numpy array with shape (n_s, n_s)
            D (2d numpy array): A numpy array with shape (n_s, n_a)
            E (2d numpy array): A numpy array with shape (n_s, n_a)
            f (2d numpy array): A numpy array with shape (n_s,1)
            g (2d numpy array): A numpy array with shape (n_a,1)
            h (1d numpy array): A numpy array with shape (1,)

            where the following equation should hold
            Q_t^*(s) = s^T C s + a^T D s + s^T E a + f^T s  + g^T a + h

        """
        C = Q + (A.T @ P @ A)
        D = R + (B.T @ P @ B)
        E = M + (2 * (A.T @ P @ B))
        ft = q.T + (2 * (m.T @ P @ A)) + (y.T @ A)
        gt = r.T + (2 * (m.T @ P @ B)) + (y.T @ B)
        h = (m.T @ P @ m) + (y.T @ m + p) + b

        return C, D, E, ft.T, gt.T, h.flatten()

    def compute_policy(self, A, B, m, C, D, E, f, g, h):
        """
        Compute the optimal policy at the current time step t
        Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)


        Let Q_t^*(s) = s^T C s + a^T D a + s^T E a + f^T s  + g^T a  + h
        Parameters:
            A (2d numpy array): A numpy array with shape (n_s, n_s)
            B (2d numpy array): A numpy array with shape (n_s, n_a)
            m (2d numpy array): A numpy array with shape (n_s, 1)
            C (2d numpy array): A numpy array with shape (n_s, n_s). C is PD.
            D (2d numpy array): A numpy array with shape (n_a, n_a). D is PD.
            E (2d numpy array): A numpy array with shape (n_s, n_a)
            f (2d numpy array): A numpy array with shape (n_s, 1)
            g (2d numpy array): A numpy array with shape (n_a, 1)
            h (1d numpy array): A numpy array with shape (1, )
        Returns:
            K_t (2d numpy array): A numpy array with shape (n_a, n_s)
            k_t (2d numpy array): A numpy array with shape (n_a, 1)

            where the following holds
            \pi*_t(s) = K_t s + k_t
        """
        D_inv = np.linalg.inv(D)
        K = -0.5 * (D_inv @ E.T)
        k = -0.5 * (D_inv @ g)
        return K, k.flatten()

    def compute_V_params(self, A, B, m, C, D, E, f, g, h, K, k):
        """
        Compute the V function parameters for the next time step
        Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
        Let V_t^*(s) = s^TP_ts + y_t^Ts + p_t
        Parameters:
            A (2d numpy array): A numpy array with shape (n_s, n_s)
            B (2d numpy array): A numpy array with shape (n_s, n_a)
            m (2d numpy array): A numpy array with shape (n_s, 1)
            C (2d numpy array): A numpy array with shape (n_s, n_s). C is PD.
            D (2d numpy array): A numpy array with shape (n_a, n_a). D is PD.
            E (2d numpy array): A numpy array with shape (n_s, n_a)
            f (2d numpy array): A numpy array with shape (n_s, 1)
            g (2d numpy array): A numpy array with shape (n_a, 1)
            h (1d numpy array): A numpy array with shape (1, )
            K (2d numpy array): A numpy array with shape (n_a, n_s)
            k (2d numpy array): A numpy array with shape (n_a, 1)

        Returns:
            P_t (2d numpy array): A numpy array with shape (n_s, n_s)
            y_t (2d numpy array): A numpy array with shape (n_s, 1)
            p_t (1d numpy array): A numpy array with shape (1,)
        """
        P = C + (K.T @ D @ K) + (E @ K)
        yt = (2 * (k.T @ D @ K)) + (k.T @ E.T) + (g.T @ K) + f.T
        p = (k.T @ D @ k) + (g.T @ k) + h

        return P, yt.T, p.flatten()

    def lqr(self, s_star, a_star, T):
        """
        Compute optimal policies by solving
        argmin_{\pi_0,...\pi_{T-1}} \sum_{t=0}^{T-1} s_t^T Q s_t + a_t^T R a_t + s_t^T M a_t + q^T s_t + r^T a_t
        subject to s_{t+1} = A s_t + B a_t + m, a_t = \pi_t(s_t)

        Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
        Let optimal \pi*_t(s) = K_t s + k_t

        Parameters:
        s_star (numpy array) with shape (4,)
        a_star (numpy array) with shape (1,)
        T (int): The number of total steps in finite horizon settings

        Returns:
            ret (list): A list, [(K_0, k_0), (K_1, k_1), ..., (K_{T-1}, k_{T-1})]
            and the shape of K_t is (n_a, n_s), the shape of k_t is (n_a,)
        """
        #TODO
        N_s = s_star.shape[0]
        N_a = a_star.shape[0]

        # get A, B, Q, R, M, q, r
        A, B, Q, R, M, q, r = self.local_controller.compute_local_expansions(s_star, a_star)

        # Create H block and make PD
        H = np.block([[Q, M], [M.T, R]])
        num_reg = 1e-7
        spectrum, eigvec = np.linalg.eig(H)
        H_new = np.zeros(H.shape)
        for i, sigma in enumerate(spectrum):
            if sigma > 0:
                H_new += sigma * eigvec[:, i:i + 1] @ eigvec[:, i:i + 1].T
        H_new += num_reg * np.identity(H.shape[0])
        Q_new = H_new[:N_s, :N_s]
        R_new = H_new[N_s:, N_s:]
        M_new = H_new[:N_s, N_s:]
        # Extract updated Q_2, M_2, R_2, q_2, r_2
        Q_2 = Q_new / 2
        R_2 = R_new / 2
        q_2 = (q.T - s_star.T @ Q - a_star.T @ M.T).T
        r_2 = (r.T - a_star.T @ R - s_star.T @ M).T
        b = self.local_controller.c(s_star, a_star) + 1 / 2 * s_star.T @ Q_2 @ s_star + 1 / 2 * a_star.T @ R_2 @ a_star + s_star.T @ M @ a_star - q.T @ s_star - r.T @ a_star
        m = self.local_controller.f(s_star, a_star) - A @ s_star - B @ a_star

        # q_2 = q_2[:, None]
        # r_2 = r_2[:, None]
        # b = b.flatten()
        # m = m[:, None]

        # Initialize P, y, p for value function iterations (backwards recursion)
        P = np.zeros((N_s, N_s))  # terminal cost is 0, so P is initialized to 0
        y = np.zeros((N_s, 1))
        p = 0 
        # Compute K, k with base step time t = T-1
        policies = []
        for t in range(T-1, -1, -1):
            # Compute the Q function parameters for time step t
            C, D, E, f, g, h = self.compute_Q_params(A, B, Q_2, R_2, M_new, q_2, r_2, m, b, P, y, p)
            
            # Compute the optimal policy at the current time step t
            K, k = self.compute_policy(A, B, m, C, D, E, f, g, h)
            
            # Compute the V function parameters for the next time step
            P, y, p = self.compute_V_params(A, B, m, C, D, E, f, g, h, K, k)
            
            policies.insert(0, (K, k))  # Insert at the beginning to maintain correct order

        # return policy
        return policies
        pass

