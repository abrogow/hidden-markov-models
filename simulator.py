import licensePlateDB as lpdb
import matplotlib.pyplot as plt
import numpy as np
import string
import argparse
import yaml


def generate_random_license_plate(alphabet, length):
    return "".join(np.random.choice(alphabet, length))


def generate_noisy_measurement(license_plate, alphabet, interference_model):
    noisy_measurement = ''
    for char in license_plate:
        char_idx = alphabet.index(char)
        interference_model_row = interference_model[char_idx, :]
        new_char_idx = np.where(np.random.rand() < np.cumsum(interference_model_row))[0][0]
        new_char = alphabet[new_char_idx]
        noisy_measurement += new_char
    return noisy_measurement


DEFAULT_CFG_PATH = "config/default.yml"


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="License Plate DB tester"
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CFG_PATH, dest="config_path",
                        help="Path to YAML file with test configuration")
    argv = parser.parse_args()

    return argv


def get_configuration(args):
    # Load configuration from YAML file
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_simulation(
    m,  # Length of the alphabet
    n,  # Length of the license plate number
    interference_model,
    num_of_license_plates_in_db,
    num_of_license_plates_to_test,
):

    alphabet = [x for x in (string.ascii_uppercase + string.digits)[:m]]

    db = lpdb.LicensePlateDB(interference_model, n, alphabet)

    license_plates = []

    for i in range(0, num_of_license_plates_in_db):

        while True:
            # Add random license plate
            lp = generate_random_license_plate(alphabet, n)
            recognized_as = generate_noisy_measurement(lp, alphabet, interference_model)

            # Continue only if this is not a duplicate
            duplicate_found = False
            for x in license_plates:
                if lp == x[0] or recognized_as == x[1]:
                    duplicate_found = True
            if not duplicate_found:
                break

        license_plates.append([lp, recognized_as])
        db.add(recognized_as)

    frr = np.zeros(num_of_license_plates_to_test)
    far = np.zeros(num_of_license_plates_to_test)

    for i in range(0, num_of_license_plates_to_test):

        print(f"{i}/{num_of_license_plates_to_test}", end='\r')

        # Add random license plate
        lp = generate_random_license_plate(alphabet, n)
        recognized_as = generate_noisy_measurement(lp, alphabet, interference_model)
        license_plates.append([lp, recognized_as])
        db.add(recognized_as)

        # Choose random license plate to remove
        lp_idx = int(np.random.rand() * len(license_plates))
        lp = license_plates[lp_idx][0]
        initially_recognized_as = license_plates[lp_idx][1]

        # Remove random license plate
        noisy_lp = generate_noisy_measurement(lp, alphabet, interference_model)
        recognized_as, was_rejected = db.remove(noisy_lp)

        to_remove = None
        for idx, x in enumerate(license_plates):
            if recognized_as == x[1]:
                to_remove = idx
        license_plates.pop(to_remove)

        # Update statistics
        if i > 0:
            frr[i] = frr[i - 1]
            far[i] = far[i - 1]
        if was_rejected:
            frr[i] += 1
        if not was_rejected and (initially_recognized_as != recognized_as):
            far[i] += 1

    far = far/np.arange(1, len(far) + 1)
    frr = far/np.arange(1, len(far) + 1)

    return far, frr


def plot_far(far):
    plt.plot(far)
    plt.title("False Acceptance Rate")
    plt.ylabel("False Acceptance Rate")
    plt.xlabel("Iteration number")
    plt.grid(True)
    plt.show()


def main():
    args = parse_args()

    print("Loading configuration...")
    config = get_configuration(args)

    print("Running simulation...")
    far, frr = run_simulation(
        config['length_of_the_alphabet'],
        config['length_of_license_plate'],
        np.array(config['confusion_matrix']),
        config['num_of_license_plates_in_db'],
        config['num_of_license_plates_to_test'],
    )

    print("Plotting results...")
    plot_far(far)


if __name__ == "__main__":
    main()