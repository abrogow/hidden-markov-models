import numpy as np
from hmm import HiddenMarkovModel


class Viterbi:

    def __init__(self, hmm):

        if not isinstance(hmm, HiddenMarkovModel):
            raise TypeError("hmm must be a HiddenMarkovModel")

        self.hmm = hmm

    def calculate_optimal_state_sequence(self, observation_sequence):
        """
        Given the observation sequence calculates optimal state sequence
        :param observation_sequence: Observation sequence
        :return: optimal state sequence
        """
        A = self.hmm.transition_pd
        B = self.hmm.observation_pd
        pi = self.hmm.initial_state_pd
        N = self.hmm.numOfStates
        M = self.hmm.numOfObservations
        T = len(observation_sequence)

        if np.max(observation_sequence) >= M:
            raise RuntimeError("Invalid observation sequence")

        sigma = np.zeros([N, T])
        phi = np.zeros([N, T]) # Argument maximizing sigma

        # Initialization
        for i in range(0, N):
            sigma[i, 0] = pi[i] * B[i, observation_sequence[0]]

        # Recursion
        # First iterate over time
        for t in range(1, T):
            # Iterate over states
            for j in range(0, N):
                tmp = np.multiply(sigma[:, t-1] , A[:, j])
                sigma[j, t] = np.max(tmp * B[j, observation_sequence[t]])
                phi[j, t] = np.argmax(tmp)

        q = np.zeros_like(observation_sequence)

        # Termination
        rho = np.max(sigma[:, T-1])
        q[T-1] = np.argmax(sigma[:, T-1])

        # Path backtracing
        for t in range(T-1, 0, -1):
            q[t-1] = phi[q[t], t]

        return q, sigma




