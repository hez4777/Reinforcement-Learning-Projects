from pa2.finite_difference_method import gradient, hessian
import numpy as np
from pa2.local_controller import LocalController
from tqdm import tqdm

class DDPController:
    def __init__(self, env):
        self.env = env
        self.pred_time = 50
        self.N_s = env.observation_space.shape[0]
        self.umax = 20 # constraint on actions
        self.local_controller = LocalController(env)

        self.v = [0.0 for _ in range(self.pred_time + 1)]
        self.v_x = [np.zeros(self.N_s) for _ in range(self.pred_time + 1)]
        self.v_xx = [np.zeros((self.N_s, self.N_s)) for _ in range(self.pred_time + 1)]
        
        self.f = lambda x,u : env._state_eq(x,u) # dynamic
        self.l = lambda x, u: 0.5 * np.sum(np.square(u))  # l(x, u)
        self.lf = lambda x: 0.5 * (np.square(1.0 - np.cos(x[2])) + np.square(x[1]) + np.square(x[3])) # final cost
        
        # Cost
        # expansion final cost
        c_constant_u = lambda x: self.lf(x)
        self.lf_x = lambda s: gradient(c_constant_u, s)
        self.lf_xx = lambda s: hessian(c_constant_u, s)
        # from your implementation
        self.l_x = lambda s,a: self.local_controller.compute_local_expansions(s,a, self.f, self.l)[5]
        self.l_u = lambda s,a: self.local_controller.compute_local_expansions(s,a, self.f, self.l)[6]
        self.l_xx = lambda s,a: self.local_controller.compute_local_expansions(s,a, self.f, self.l)[2]
        self.l_uu = lambda s,a: self.local_controller.compute_local_expansions(s,a, self.f, self.l)[3]
        self.l_ux = lambda s,a: self.local_controller.compute_local_expansions(s,a, self.f, self.l)[4]
        
        # Dynamic
        # from your implementation
        self.f_x = lambda s,a: self.local_controller.compute_local_expansions(s,a,self.f, self.l)[0]
        self.f_u = lambda s,a: self.local_controller.compute_local_expansions(s,a,self.f, self.l)[1]
        # expansion dynamics jacobian of jacobian 
        self.f_xx = lambda s,a: jacobian_tensor(self.f_x, s, a, 0) # |S|x|S|x|S|
        self.f_uu = lambda s,a: jacobian_tensor(self.f_u, s, a, 1) # |S|x|A|x|A|
        self.f_ux = lambda s,a: jacobian_tensor(self.f_u, s, a, 0) # |S|x|A|x|S|

    def backward(self, x_seq, u_seq):
        """
        Compute optimal policies given sequence of states and actions.

        Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
        Let optimal \pi*_t(s) = K_t s + k_t

        Parameters:
        x_seq (numpy array) with shape (T,n_s)
        u_seq (numpy array) with shape (T,n_a)

        Returns:
            Two lists, [k_0, k_1, ..., k_{T-1}] and [K_0, K_1, ..., K_{T-1}]
            and the shape of k_t is (n_a,), the shape of K_t is (n_a, n_s)
        """

        # Cost at T
        self.v[-1] = self.lf(x_seq[-1]) # cost_T
        self.v_x[-1] = self.lf_x(x_seq[-1]) # q_T
        self.v_xx[-1] = self.lf_xx(x_seq[-1]) # Q_T
        
        # Policy
        k_seq = []
        kk_seq = []

        for t in range(self.pred_time - 1, -1, -1):
            f_x_t = self.f_x(x_seq[t], u_seq[t]) # A_t 
            f_u_t = self.f_u(x_seq[t], u_seq[t]) # B_t
            q_x = self.l_x(x_seq[t], u_seq[t]) + np.matmul(f_x_t.T, self.v_x[t + 1]) # q_t + A_t^T @ q'_{t+1}
            q_u = self.l_u(x_seq[t], u_seq[t]) + np.matmul(f_u_t.T, self.v_x[t + 1]) # r_t + B_t^T @ q'_{t+1}
            q_xx = self.l_xx(x_seq[t], u_seq[t]) + \
              np.matmul(np.matmul(f_x_t.T, self.v_xx[t + 1]), f_x_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_xx(x_seq[t], u_seq[t]))) # Q_t + A_t^T @ Q_{t+1} @ A_t + q_{t+1}^T @ X_t
            tmp = np.matmul(f_u_t.T, self.v_xx[t + 1]) # B_t^T @ Q_{t+1} |A|x|S|
            q_uu = self.l_uu(x_seq[t], u_seq[t]) + np.matmul(tmp, f_u_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_uu(x_seq[t], u_seq[t]))) # R_t + B_t^T @ Q_{t+1} @ B_t + q_{t+1}^T @ Y_t
            q_ux = self.l_ux(x_seq[t], u_seq[t]).T + np.matmul(tmp, f_x_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_ux(x_seq[t], u_seq[t]))) # M_t + B_t^T @ Q_{t+1} @ A_t + q_{t+1}^T @ Z_t
            inv_q_uu = np.linalg.inv(q_uu)
            k = -np.matmul(inv_q_uu, q_u) 
            kk = -np.matmul(inv_q_uu, q_ux) 
            dv = 0.5 * np.matmul(q_u, k) 
            self.v[t] += dv
            self.v_x[t] = q_x - np.matmul(np.matmul(q_u, inv_q_uu), q_ux)
            self.v_xx[t] = q_xx + np.matmul(q_ux.T, kk)
            k_seq.append(k)
            kk_seq.append(kk)
        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        """
        Rolling out a new trajectory given an observed sequence of 
        states and actions, and policy.

        Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)

        Parameters:
        x_seq (numpy array) with shape (T+1,n_s)
        u_seq (numpy array) with shape (T,n_a)
        k_seq (numpy array) with shape (T,n_a)
        kk_seq (numpy array) with shape (T,n_a,n_s)

        Returns:
            Two lists, x_seq_hat and u_seq_hat that represent new rollout
            states and actions sequence. The shape of x_seq_hat is (T+1,n_s)
            and the shape of u_seq_hat is (T,n_a)
        """
        # TODO
        # initialize s_0, a_0

        # for t=0,...,T-1 calculate x_hat_t and u_hat_t, i.e., the states sequence
        # and the actions sequence
        # Note: please clip the action within (-self.umax, self.umax), you can use np.clip

        # return x_seq_hat and u_seq_hat
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        for t in range(len(u_seq)):
            action = k_seq[t] + np.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
            u_seq_hat[t] = np.clip(u_seq[t] + action, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])
        return x_seq_hat, u_seq_hat
        pass

    def clip_action(self, action):
        """
        Clips action to stay within the environment's allowed range.

    Parameters:
    action (numpy array) with shape (n_a,)

    Returns:
    A clipped action array within the valid range.
    """
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action
    
    def train(self, init_obs):
        """
        Given initial observation of a state trajectory, calculate optimal policy.

        Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)

        Parameters:
        init_obs (numpy array) with shape (n_s,)

        Returns:
            A list, actions that represent the optimal action at each 
            timestep. The shape of actions is (# steps,n_a) (now we set # steps=400)
        """
        u_seq = [np.zeros(1) for _ in range(self.pred_time)]
        x_seq = [init_obs.copy()]
        for t in range(self.pred_time):
            x_seq.append(self.env._state_eq(x_seq[-1], u_seq[t]))

        policys = []
        for i in tqdm(range(400)):
            # TODO
            # backward to calculate k_seq and kk_seq
            k_seq, kk_seq = self.backward(x_seq, u_seq)
            # forward to calculate new x_seq and u_seq
            x_seq, u_seq = self.forward(x_seq, u_seq, k_seq, kk_seq)
            # step the environment with the policy for s_0
            action = np.clip(u_seq[0], -self.umax, self.umax)
            state, _, _, _ = self.env.step(action)
            # redefine the s_0 to be the next observation
            x_seq[0] = state.copy()
            # append the policy for s_0 to policys
            policys.append(action)
            # pass
        return policys

    

def jacobian_tensor(f, x, a, index, delta=1e-5):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts 2D numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 3D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    m,n = f(x,a).shape
    x = x.astype('float64')
    if index==0:
        l, = x.shape
        jacobian = np.zeros((m, n, l)).astype('float64')
        for i in range(n):
            x[i] += delta
            gplus = f(x,a)
            x[i] -= 2 * delta
            gminus = f(x,a)
            x[i] += delta
        jacobian[:, i] = (gplus - gminus) / (2 * delta)

    else:
        l, = a.shape
        jacobian = np.zeros((m, n, l)).astype('float64')
        for i in range(n):
            a[i] += delta
            gplus = f(x,a)
            a[i] -= 2 * delta
            gminus = f(x,a)
            a[i] += delta
        jacobian[:, i] = (gplus - gminus) / (2 * delta)
    
    return jacobian