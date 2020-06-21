import hmm
import numpy as np


def test_hmm_viterbi_sigma():
    A = np.array([[0.7, 0.3], [0.6, 0.4]])
    B = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
    pi = np.array([0.6, 0.4])
    model = hmm.HiddenMarkovModel(A, B, pi)

    O = [1, 2] # observation sequence
    viterbi = hmm.Viterbi(model)
    _, sigma = viterbi.calculate_optimal_state_sequence(O)

    sigma_reference = np.array([[0.12, 0.0588], [0.12, 0.0096]])
    np.testing.assert_allclose(sigma, sigma_reference)


def test_hmm_viterbi_state_sequence():
    A = np.array([[0.5, 0.5], [0.5, 0.5]])
    B = np.array([[0.45, 0.45, 0.1], [0.1, 0.1, 0.8]])
    pi = np.array([0.5, 0.5])
    model = hmm.HiddenMarkovModel(A, B, pi)

    O = [0, 1, 2, 0, 2] # observation sequence
    viterbi = hmm.Viterbi(model)
    q, _ = viterbi.calculate_optimal_state_sequence(O)

    q_reference = np.array([0, 0, 1, 0, 1])
    np.testing.assert_allclose(q, q_reference)