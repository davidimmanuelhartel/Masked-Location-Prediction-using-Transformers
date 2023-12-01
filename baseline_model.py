import torch

def baseline_model(masked_tokens, true_tokens, highest_rank_pos_labels, ntokens):
    """
    Fills masked positions in masked_tokens with probabilities corresponding to the highest rank position label.

    :param masked_tokens: Tensor containing masked tokens.
    :param true_tokens: Tensor containing the true tokens. This is only used to identify masked positions.
    :param highest_rank_pos_label: The index of the label in the embedding space.
    :param ntokens: The size of the embedding space.
    :return: Tensor with probabilities corresponding to the highest rank position label for masked positions.
    """
    batch_size, sequence_length = masked_tokens.size()
    
    # Initialize a tensor to hold the probabilities
    predicted_probs = torch.zeros(batch_size, sequence_length, ntokens)

    # Find positions where masked_tokens differ from true_tokens
    masked_positions = masked_tokens != true_tokens


    # Fill in the masked positions with the label probabilities
    for i in range(batch_size):
        # Create a one-hot encoded representation of the highest rank position label for each sequence
        label_probs = torch.zeros(ntokens)
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