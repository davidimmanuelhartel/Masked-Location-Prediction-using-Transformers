import torch

class BaselineModel:
    def __init__(self, dataset):
        """
        Initialize the BaselineModel class with tokens, highest rank position labels, and the size of the embedding space.
        
        :param dataset: The dataset to use for the model.
        """
        self.highest_rank_dict = dataset.get_highest_rank_pos_dict()
        self.dataset = dataset
        self.ntokens = len(dataset.vocab)

    
    def predict(self, masked_tokens, true_tokens, user_weeks):
        """
        Fills masked positions in masked_tokens with probabilities corresponding to the highest rank position label.

        :param masked_tokens: Tensor containing masked tokens.
        :param true_tokens: Tensor containing the true tokens. This is only used to identify masked positions.
        :return: Tensor with probabilities corresponding to the highest rank position label for masked positions.
        """
        batch_size, sequence_length = masked_tokens.size()
        
        # Initialize a tensor to hold the probabilities
        predicted_probs = torch.zeros(batch_size, sequence_length, self.ntokens)

        # Find positions where masked_tokens differ from true_tokens
        masked_positions = masked_tokens != true_tokens
        
        # create list of highest rank positions for each user_week in batch
        highest_rank_pos = [self.highest_rank_dict[i] for i in user_weeks]
        highest_rank_pos_labels = self.dataset.encode_positions(highest_rank_pos)
        
        # Fill in the masked positions with the label probabilities
        for i in range(batch_size):
            # Create a one-hot encoded representation of the highest rank position label for each sequence
            label_probs = torch.zeros(self.ntokens)
            label_probs[highest_rank_pos_labels[i]] = 1
            for j in range(sequence_length):
                if masked_positions[i, j]:
                    predicted_probs[i, j] = label_probs
                else:
                    # For unmasked positions, you might want to keep the original probabilities
                    # For simplicity, assuming a one-hot encoding for the actual token
                    # Modify this part according to your actual scenario
                    token = masked_tokens[i, j]
                    predicted_probs[i, j, token] = 1

        return predicted_probs