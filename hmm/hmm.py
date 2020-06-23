import numpy as np


def verify_probability_distribution(pd):

    if len(pd.shape) == 1 and not np.allclose(np.sum(pd), 1):
        raise RuntimeError("Invalid single dimensional probability distribution")

    elif len(pd.shape) == 2:
        for idx, row in enumerate(pd):
            sum = np.sum(row)
            if not np.allclose(sum, 1):
                raise RuntimeError("Invalid two dimensional probability distribution"
                                   f"(sum of the row numer {idx} is {sum})")


class HiddenMarkovModel:

    def __init__(self, transition_pd, observation_pd, initial_state_pd):
        """
        :param transition_pd: state transition probability distribiution
        :param observation_pd: observation symbol probability distribiution
        :param initial_state_pd: initial state probability distribiution
        """
        self.transition_pd = transition_pd
        self.observation_pd = observation_pd
        self.initial_state_pd = initial_state_pd

        self.verify_shapes()

        # verify_probability_distribution(transition_pd)
        verify_probability_distribution(observation_pd)
        verify_probability_distribution(initial_state_pd)

        self.numOfStates = transition_pd.shape[0]

        # In this case we assume that number of observatons is the same as number of states anyway
        self.numOfObservations = observation_pd.shape[1]

    def verify_shapes(self):

        if len(self.transition_pd.shape) != 2:
            raise RuntimeError("Transition Probability Distribution has to be a matrix")

        if self.transition_pd.shape[0] != self.observation_pd.shape[0]:
            raise RuntimeError("Transition Probability Distribution and Observation Probability Distribution have "
                               "to have the same length of the 1st dimension")

        if self.transition_pd.shape[0] != self.transition_pd.shape[1]:
            raise RuntimeError("Transition Probability Distribution has to be a square matrix")

        if len(self.initial_state_pd.shape) != 1:
            raise RuntimeError("Initial State Probability Distribution has to be a vector")

        if self.transition_pd.shape[0] != self.initial_state_pd.shape[0]:
            raise RuntimeError("Invalid lenght of Initial State Probability Distribution")

    def __str__(self):
        string = '\nTransition Probability Distribution:\n'
        string += str(self.transition_pd)
        string += '\nObservation Probability Distribution:\n'
        string += str(self.observation_pd)
        string += '\nInitial State Probability Distribution:\n'
        string += str(self.initial_state_pd)
        return string
