import numpy as np

class MDP:
    def __init__(self, P, R, gamma=0.9):
        """
        P: numpy array of shape (nS, nS), row-normalized transition probabilities
        R: numpy array of shape (nS,) for state-based rewards or (nS, nS) for transition-based rewards
        gamma: discount factor
        """
        self.P = np.array(P, dtype=float)
        # Ensure each row sums to 1
        if not np.allclose(self.P.sum(axis=1), 1.0):
            raise ValueError("Each row of P must sum to 1.")
        self.nS = self.P.shape[0]
        self.gamma = gamma

        R = np.array(R, dtype=float)
        if R.ndim == 1:
            if R.shape[0] != self.nS:
                raise ValueError("When R is 1D, its length must equal nS.")
            # Broadcast state-rewards to a full matrix
            self.R = np.tile(R[:, None], (1, self.nS))
        elif R.ndim == 2:
            if R.shape != (self.nS, self.nS):
                raise ValueError("When R is 2D, its shape must be (nS, nS).")
            self.R = R
        else:
            raise ValueError("R must be either a 1D or 2D array.")

def value_iteration(mdp, theta=1e-6):
    """
    Perform value iteration on an MDP with a single implicit action per state.
    Returns the optimal value function V of shape (nS,).
    """
    V = np.zeros(mdp.nS)
    while True:
        delta = 0.0
        V_new = np.zeros_like(V)
        for s in range(mdp.nS):
            # Bellman update for state s:
            # V_new[s] = sum_{s'} P[s, s'] * (R[s, s'] + gamma * V[s'])
            V_new[s] = np.sum(mdp.P[s] * (mdp.R[s] + mdp.gamma * V))
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < theta:
            break
    return V

