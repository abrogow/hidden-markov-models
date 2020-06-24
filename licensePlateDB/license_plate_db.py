import hmm
import numpy as np


class LicensePlateDB:

    def __init__(self, interference_model, license_plate_number_len, alphabet):
        """
        :param interference_model: interference model
        :param license_plate_number_len: length of license plate number
        :param alphabet: list of allowed characters in license plate
        """
        self.interference_model = interference_model
        self.license_plate_number_len = license_plate_number_len
        self.alphabet = alphabet
        self.extended_alphabet = alphabet + ['$', '#']
        self.license_plates = []
        self.hmm = None
        self.viterbi = None

        self.verify_init_params()

    def add(self, license_plate):
        """
        Function add new license plate to Hidden Markov Model
        :param license_plate:
        :return:
        """
        self.verify_license_plate(license_plate)

        # Add licence plate to the list and regenerate HMM
        self.license_plates.append(license_plate)
        self._generate_hidden_markov_model()

    def remove(self, license_plate):
        """
        Function recognize if license plate is the same as recognized.
        If yes, removes it from the model.
        If no, reject it.
        :param license_plate:
        :return:recognized_license_plate, was_rejected
        """
        self.verify_license_plate(license_plate)

        # Convert license plate number to list of indices in the alphabet
        license_plate_idxs = self._convert_to_idx_vector(license_plate)

        # Find the most probable sequence of states
        state_sequence, _ = self.viterbi.calculate_optimal_state_sequence(license_plate_idxs)

        # Convert back to a string
        recognized_license_plate = ''
        state_sequence = state_sequence[1:-1]  # Drop start and end markers
        for state_idx in state_sequence:
            license_plate_idx = int(state_idx/self.license_plate_number_len)
            pos = state_idx % self.license_plate_number_len
            recognized_license_plate += self.license_plates[license_plate_idx][pos]

        if recognized_license_plate in self.license_plates:
            self.license_plates.remove(recognized_license_plate)
            self._generate_hidden_markov_model()
            was_rejected = False
        else:
            was_rejected = True

        return recognized_license_plate, was_rejected

    def verify_init_params(self):

        if self.interference_model.shape[0] != self.interference_model.shape[1]:
            raise RuntimeError("Interference Model has to be a square matrix"
                               f"(is ({self.interference_model.shape[0], self.interference_model.shape[1]}) instead)")

        if self.interference_model.shape[0] != len(self.alphabet):
            raise RuntimeError("Alphabet has to have the same length as Interference Model 1st dimension")

    def verify_license_plate(self, license_plate):

        if len(license_plate) != self.license_plate_number_len:
            raise RuntimeError("License plate has wrong length")

        for i in range(0, len(license_plate) - 1):
            if license_plate[i] not in self.alphabet:
                raise RuntimeError("Invalid character in license plate")

    def _generate_transition_pd(self, num_of_states):

        # Transition Probability Distribution
        transition_pd = np.zeros([num_of_states] * 2, dtype=float)
        for license_plate_idx, license_plate in enumerate(self.license_plates):
            for pos, char in enumerate(license_plate):
                curr_state_idx = license_plate_idx * self.license_plate_number_len + pos
                if pos == (len(license_plate) - 1):
                    transition_pd[curr_state_idx, -1] += 1
                else:
                    # Transition from current to next character
                    transition_pd[curr_state_idx, curr_state_idx + 1] += 1

                    # If first character, add transition from start marker to it
                    if pos == 0:
                        transition_pd[-2, curr_state_idx] += 1

        # Normalize probabilities in Transition Probability Distribution matrix
        for i in range(0, transition_pd.shape[0]):
            row = transition_pd[i, :]
            if not np.allclose(np.sum(row), 0.0):
                transition_pd[i, :] = row / np.sum(row)

        return transition_pd

    def _generate_observation_pd(self, num_of_states, num_of_observations):

        observation_pd = np.zeros([num_of_states, num_of_observations])
        observation_pd[-2, -2] = 1.0  # In the start state we always get beginning marker
        observation_pd[-1, -1] = 1.0  # In the end state we always get end marker

        for license_plate_idx, license_plate in enumerate(self.license_plates):
            for pos, char in enumerate(license_plate):
                curr_state_idx = license_plate_idx * self.license_plate_number_len + pos
                observation_pd[curr_state_idx, :-2] = self.interference_model[self.alphabet.index(char), :]

        return observation_pd

    def _generate_intial_state_pd(self, num_of_states):

        initial_state_pd = np.zeros(num_of_states)
        initial_state_pd[-2] = 1.0  # We always start from beginning marker

        return initial_state_pd

    def _generate_hidden_markov_model(self):

        # Calculate number of states
        num_of_states = len(self.license_plates) * self.license_plate_number_len + 2

        # We can observe every character of the alphabet as well as beginning and end markers.
        num_of_observations = len(self.alphabet) + 2

        transition_pd = self._generate_transition_pd(num_of_states)
        observation_pd = self._generate_observation_pd(num_of_states, num_of_observations)
        initial_state_pd = self._generate_intial_state_pd(num_of_states)

        self.hmm = hmm.HiddenMarkovModel(transition_pd, observation_pd, initial_state_pd)
        self.viterbi = hmm.Viterbi(self.hmm)

    def _convert_to_idx_vector(self, license_plate):
        """
        Function convert license plate string to vector of alphabet indexes
        :param license_plate:
        :return: license_plate_idxs
        """
        license_plate = '$' + license_plate + '#'  # Add beginning and end markers
        return [self.extended_alphabet.index(x) for x in license_plate]
