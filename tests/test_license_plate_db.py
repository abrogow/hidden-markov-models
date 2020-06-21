import licensePlateDB as lpdb
import numpy as np
import random


def test_lpdb_hmm_generation():

    alphabet = ['a', 'b', 'c']
    n = 3 # License plate number length
    interference_model = np.eye(3)

    db = lpdb.LicensePlateDB(interference_model, n, alphabet)

    db.add("abc")
    db.add("acc")
    db.add("abb")

    # Reference Transition Probability Distribution
    ref_transition_pd = np.zeros([11, 11])
    ref_transition_pd[0, 4] = 2/3
    ref_transition_pd[0, 5] = 1/3
    ref_transition_pd[4, 7] = 1/2
    ref_transition_pd[4, 8] = 1/2
    ref_transition_pd[5, 8] = 1.0
    ref_transition_pd[7, 10] = 1.0
    ref_transition_pd[8, 10] = 1.0
    ref_transition_pd[9, 0] = 1.0

    # Reference Observation Symbol Probability Distribution
    ref_observation_pd = np.zeros([11, 5])
    ref_observation_pd[:-2, :-2] = np.tile(np.eye(3), (3, 1))
    ref_observation_pd[-2, -2] = 1.0  # In the start state we always get beginning marker
    ref_observation_pd[-1, -1] = 1.0  # In the end state we always get end marker

    # Reference Initial State Probability Distribution
    ref_initial_state_pd = np.zeros(11)
    ref_initial_state_pd[-2] = 1.0

    np.testing.assert_allclose(db.hmm.transition_pd, ref_transition_pd)
    np.testing.assert_allclose(db.hmm.observation_pd, ref_observation_pd)
    np.testing.assert_allclose(db.hmm.initial_state_pd, ref_initial_state_pd)


def test_lpdb_removal_perfect_measurements():

    alphabet = ['a', 'b', 'c']
    n = 20  # License plate number length
    interference_model = np.eye(3)

    db = lpdb.LicensePlateDB(interference_model, n, alphabet)

    num_of_license_plates = 20
    license_plates = []

    # Generate random license plates and add to DB
    for i in range(0, num_of_license_plates):
        lp = "".join(np.random.choice(alphabet, n))
        license_plates.append(lp)
        db.add(lp)

    # Shuffle list of license plates
    random.shuffle(license_plates)

    # Remove license plates one by one
    for lp in license_plates:
        recognized_as, was_rejected = db.remove(lp)
        assert(not was_rejected)
        assert(recognized_as == lp)


def generate_noisy_measurement(license_plate, alphabet, interference_model):
    noisy_measurement = ''
    for char in license_plate:
        char_idx = alphabet.index(char)
        interference_model_row = interference_model[char_idx, :]
        new_char_idx = np.where(np.random.rand() < np.cumsum(interference_model_row))[0][0]
        new_char = alphabet[new_char_idx]
        noisy_measurement += new_char
    return noisy_measurement


def test_lpdb_removal_noisy_measurements():

    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    n = 10  # License plate number length
    m = len(alphabet)
    confusion_probability = 0.02
    interference_model = np.eye(m) * (1.0 - m * confusion_probability) + np.ones([m] * 2) * confusion_probability

    db = lpdb.LicensePlateDB(interference_model, n, alphabet)

    num_of_license_plates = 100
    license_plates = []

    # Generate random license plates and add to DB
    for i in range(0, num_of_license_plates):
        lp = "".join(np.random.choice(alphabet, n))
        license_plates.append(lp)
        db.add(lp)

    # Shuffle list of license plates
    random.shuffle(license_plates)

    # Remove license plates one by one
    false_rejection_counter = 0
    false_acceptance_counter = 0
    for lp in license_plates:
        noisy_lp = generate_noisy_measurement(lp, alphabet, interference_model)
        recognized_as, was_rejected = db.remove(noisy_lp)
        if was_rejected:
            false_rejection_counter += 1
        if not was_rejected and (lp != recognized_as):
            false_acceptance_counter += 1
        # print("\n")
        # print(f"License plate: {lp}")
        # print(f"Measured as:   {noisy_lp}")
        # print(f"Recognized as: {recognized_as}")
        # assert(not was_rejected)
        # assert(recognized_as == lp)

    print("\n")
    print(f"FAR: {false_acceptance_counter/num_of_license_plates}")
    print(f"FRR: {false_rejection_counter/num_of_license_plates}")