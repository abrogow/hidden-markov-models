import licensePlateDB as lpdb
import numpy as np
import random


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
    n = 4  # License plate number length
    m = len(alphabet)
    confusion_probability = 0.05
    interference_model = np.eye(m) * (1.0 - m * confusion_probability) + np.ones([m] * 2) * confusion_probability

    db = lpdb.LicensePlateDB(interference_model, n, alphabet)

    num_of_license_plates = 50
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

    print("\n")
    print(f"FAR: {false_acceptance_counter/num_of_license_plates}")
    print(f"FRR: {false_rejection_counter/num_of_license_plates}")