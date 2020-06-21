import hmm
import numpy as np


def main():
    transition_pd = np.array([[0.5, 0.5], [0.5, 0.5]])
    observation_pd = np.array([[0.3, 0.2, 0.5], [0.1, 0.1, 0.8]])
    initial_state_pd = np.array([0.5, 0.5])

    model = hmm.HiddenMarkovModel(transition_pd, observation_pd, initial_state_pd)
    viterbi = hmm.Viterbi(model)

    observationSequence = np.array([0, 1, 2, 1, 2, 0])

    stateSequence, _ = viterbi.calculate_optimal_state_sequence(observationSequence)

    print(stateSequence)


if __name__ == "__main__":
    main()
