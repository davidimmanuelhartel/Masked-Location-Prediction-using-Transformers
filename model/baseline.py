import torch
import random

class BaselineModel:
    def __init__(self, dataset, ntokens):
        """
        Initialize the BaselineModel class with a dataset and the size of the embedding space.
        
        :param dataset: The dataset to use for the model.
        :param ntokens: The size of the embedding space.
        """
        self.highest_rank_dict = dataset.get_highest_rank_pos_dict()
        self.dataset = dataset
        self.ntokens = ntokens

    def predict(self, tokens, mask, users):
        """
        Predicts labels for the masked positions in the sequence.

        :param tokens: Tensor containing the token sequence.
        :param mask: Tensor of the same shape as tokens, where True indicates a position to predict.
        :param users: List or tensor indicating the user for each sequence in the batch.
        :return: Tensor containing the predicted labels for masked positions.
        """
        batch_size, sequence_length = tokens.size()
        
        # Initialize a tensor to hold the predicted labels
        predicted_labels = torch.full((batch_size, sequence_length), -1, dtype=torch.long)  # -1 indicates no prediction

        # Create list of highest rank positions for each user_week in the batch
        highest_rank_pos = [self.highest_rank_dict[str(i) if isinstance(i, str) else str(i.item())] for i in users]
        highest_rank_pos_labels = self.dataset.encode_positions(highest_rank_pos)

        # Fill in the masked positions with the highest rank position labels
        for i in range(batch_size):
            for j in range(sequence_length):
                if mask[i, j]:
                    predicted_labels[i, j] = highest_rank_pos_labels[i]

        return predicted_labels

class BaselineModelSampled:
    def __init__(self, dataset, ntokens):
        """
        Initialize the BaselineModelSampling class with a dataset and the size of the embedding space.

        :param dataset: The dataset to use for the model.
        :param ntokens: The size of the embedding space.
        """
        self.dataset = dataset
        self.user_location_probabilities = self.dataset.calculate_location_probabilities()
        self.ntokens = ntokens

    def predict(self, tokens, mask, users):
        """
        Predicts labels for the masked positions in the sequence based on the sampling of location probabilities.

        :param tokens: Tensor containing the token sequence.
        :param mask: Tensor of the same shape as tokens, where True indicates a position to predict.
        :param users: List or tensor indicating the user for each sequence in the batch.
        :return: Tensor containing the predicted labels for masked positions.
        """
        batch_size, sequence_length = tokens.size()
        predicted_labels = torch.full((batch_size, sequence_length), -1, dtype=torch.long)

        for i in range(batch_size):
            user_id = users[i] if isinstance(users[i], str) else str(users[i].item())
            for j in range(sequence_length):
                if mask[i, j]:
                    predicted_labels[i, j] = self.sample_location(user_id)

        return predicted_labels

    def sample_location(self, user_id):
            """
            Samples a location based on the calculated probabilities for a given user.

            :param user_id: The ID of the user for whom to sample a location.
            :return: A randomly sampled location index.
            """
            user_probabilities = self.user_location_probabilities[user_id]
            locations, probabilities = zip(*user_probabilities.items())
            sampled_location = random.choices(locations, weights=probabilities, k=1)[0]
            return self.dataset.encode_positions([sampled_location])[0]

class BaselineModelMarkov1:
    def __init__(self, dataset, ntokens):
        self.dataset = dataset
        self.user_transition_probs = self.dataset.calculate_transition_probabilities()
        self.ntokens = ntokens

    def predict(self, tokens, mask, users):
        batch_size, sequence_length = tokens.size()
        predicted_labels = torch.full((batch_size, sequence_length), -1, dtype=torch.long)

        for i in range(batch_size):
            user_id = users[i] if isinstance(users[i], str) else str(users[i].item())
            user_transitions = self.user_transition_probs[user_id]
            
            for j in range(1, sequence_length):  # starting from 1 to have a previous location
                if mask[i, j]:
                    prev_location = self.dataset.decode_positions([tokens[i, j-1]])[0]
                    next_location_probs = user_transitions.get(prev_location, {})
                    # print("prev_location: ", prev_location)
                    # print("next_location_probs: ", next_location_probs)

                    if next_location_probs:
                        next_location = max(next_location_probs, key=next_location_probs.get)
                        # print("next_location: ", next_location)
                        predicted_labels[i, j] = self.dataset.encode_positions([next_location])[0]
                        # print("predicted_labels: ", predicted_labels[i, j])
                    else:
                        # Handle case with no transitions data (e.g., use the most frequent location)
                        # This part can be modified based on how you want to handle such scenarios
                        predicted_labels[i, j] = self.dataset.encode_positions([self.dataset.vocab[4]])[0]

        return predicted_labels
