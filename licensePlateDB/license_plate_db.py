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
        license_plate_idxs = self.convert_to_idx_vector(license_plate)

        # Find the most probable sequence of states
        state_sequence, _ = self.viterbi.calculate_optimal_state_sequence(license_plate_idxs)

        # Convert back to a string
        indices_in_alphabet = [self._state_idx_to_character_idx(s) for s in state_sequence]
        recognized_license_plate = self.convert_to_string_of_characters(indices_in_alphabet)

        if recognized_license_plate in self.license_plates:
            self.license_plates.remove(recognized_license_plate)
            self._generate_hidden_markov_model()
            was_rejected = False
        else:
            was_rejected = True

        return recognized_license_plate, was_rejected

    def verify_init_params(self):

        if self.interference_model.shape[0] != self.interference_model.shape[1]:
            raise RuntimeError("Interference Model has to be a square matrix")

        if self.interference_model.shape[0] != len(self.alphabet):
            raise RuntimeError("Alphabet has to have the same length as Interference Model 1st dimension")

    def verify_license_plate(self, license_plate):

        if len(license_plate) != self.license_plate_number_len:
            raise RuntimeError("License plate has wrong length")

        for i in range(0, len(license_plate) - 1):
            if license_plate[i] not in self.alphabet:
                raise RuntimeError("Invalid character in license plate")

    def _character_idx_to_state_idx(self, character_idx, position):
        return len(self.alphabet) * position + character_idx

    def _state_idx_to_character_idx(self, state_idx):
        if state_idx >= len(self.alphabet) * self.license_plate_number_len:  # Beginning and end states
            return state_idx - len(self.alphabet) * (self.license_plate_number_len - 1)
        else:
            return state_idx % len(self.alphabet)

    def _generate_transition_pd(self, num_of_states):

        # Transition Probability Distribution
        transition_pd = np.zeros([num_of_states] * 2, dtype=float)
        for license_plate in self.license_plates:
            state_sequence = self.convert_to_idx_vector(license_plate)

            # Transition from start state to first character
            first_chars_state_idx = self._character_idx_to_state_idx(state_sequence[1], 0)
            transition_pd[-2, first_chars_state_idx] += 1

            # Transition from last character to end state
            last_chars_state_idx = self._character_idx_to_state_idx(state_sequence[-2], self.license_plate_number_len - 1)
            transition_pd[last_chars_state_idx, -1] += 1

            # Iterate over remaining transitions (skipping first and last)
            for i in range(1, len(state_sequence) - 2):
                char_from_state_idx = self._character_idx_to_state_idx(state_sequence[i], i - 1)
                char_to_state_idx = self._character_idx_to_state_idx(state_sequence[i + 1], i)
                transition_pd[char_from_state_idx, char_to_state_idx] += 1

        # Normalize probabilities in Transition Probability Distribution matrix
        for i in range(0, transition_pd.shape[0]):
            row = transition_pd[i, :]
            if not np.allclose(np.sum(row), 0.0):
                transition_pd[i, :] = row / np.sum(row)

        return transition_pd

    def _generate_observation_pd(self, num_of_states, num_of_observations):

        observation_pd = np.zeros([num_of_states, num_of_observations])
        observation_pd[:-2, :-2] = np.tile(self.interference_model, (self.license_plate_number_len, 1))
        observation_pd[-2, -2] = 1.0  # In the start state we always get beginning marker
        observation_pd[-1, -1] = 1.0  # In the end state we always get end marker

        return observation_pd

    def _generate_intial_state_pd(self, num_of_states):

        initial_state_pd = np.zeros(num_of_states)
        initial_state_pd[-2] = 1.0  # We always start from beginning marker

        return initial_state_pd

    def _generate_hidden_markov_model(self):

        # Calculate number of states:
        # at each position we have a state for every character from the alphabet
        # so we multiply number of characters in the alphabet by the length of the licence plate number.
        # Then we add two states for beginning and end markers.
        num_of_states = len(self.alphabet) * self.license_plate_number_len + 2

        # We can observe every character of the alphabet as well as beginning and end markers.
        num_of_observations = len(self.alphabet) + 2

        transition_pd = self._generate_transition_pd(num_of_states)
        observation_pd = self._generate_observation_pd(num_of_states, num_of_observations)
        initial_state_pd = self._generate_intial_state_pd(num_of_states)

        self.hmm = hmm.HiddenMarkovModel(transition_pd, observation_pd, initial_state_pd)
        self.viterbi = hmm.Viterbi(self.hmm)

    def convert_to_idx_vector(self, license_plate):
        """
        Function convert license plate string to vector of alphabet indexes
        :param license_plate:
        :return: license_plate_idxs
        """
        license_plate = '$' + license_plate + '#'  # Add beginning and end markers
        return [self.extended_alphabet.index(x) for x in license_plate]

    def convert_to_string_of_characters(self, recognized_license_place_idxs):
        """
        Function convert vectof of alphabet indexes to recognized license plate string
        :param recognized_license_place_idxs:
        :return: recognized_license_plate
        """
        recognized_license_plate = ''

        for idx in recognized_license_place_idxs:
            recognized_license_plate += self.extended_alphabet[idx]

        return recognized_license_plate[1:-1]  # Strip beginning and end markers
