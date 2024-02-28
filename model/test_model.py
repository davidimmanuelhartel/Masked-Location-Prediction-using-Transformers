"""
Test Loop Functions

This script contains functions for evaluating MOBERT on test datasets.
These functions include:
- test_loop: Perform evaluation on the test set and calculate test loss and accuracy.
- test_loop_topk: Perform evaluation on the test set and calculate top-k accuracy.
"""

from fit_model import masked_loss
import torch
import build_dataset
import numpy as np


def test_loop(model, dataloader, src_vocab, iterations = 1, use_fusion_embeddings=True ):
    """
    This method performs evaluation on the test set.

    :param model: The model to be evaluated.
    :param dataloader: DataLoader for the test dataset.
    :param src_vocab: Source vocabulary size.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the model to evaluation mode.
    model.eval()
    
    # Initialize variables for total loss and accuracy.
    total_test_loss = 0
    correct_predictions = 0
    total_predictions = 0
    predicted_labels_list, true_labels_list = [], []
    accuracy_list = []
    for i in range(iterations):
        # Use torch.no_grad() for evaluation to reduce memory usage.
        with torch.no_grad():
            for X in dataloader:
                # Move tensors to the same device as the model.
                for k in X.keys():
                    if k != 'user':
                        X[k] = X[k].to(device)

                # Extract features and masked tokens.
                features = [k for k in X.keys() if k != 'y' and k != 'user']
                masked_tokens, random_mask = build_dataset.torch_mask_tokens(X['y'], src_vocab)

                # Apply mask.
                for k in features:
                    X[k][random_mask] = 0 

                # Prepare inputs for the model.
                y_input = [masked_tokens, *[X[k] for k in features]]
                y_expected = X['y']

                # Create a mask for non-padded tokens.
                src_mask = ((X['y'] != 0).int()).to(device)
                src_mask = src_mask.unsqueeze(-1)
            
                # Get model predictions.
                if use_fusion_embeddings:
                    pred = model(y_input, src_mask)
                else:
                    pred = model(masked_tokens, src_mask) # masked_tokens if without FusionEmbeddings otherwise y_input

                # Calculate loss.
                test_loss = masked_loss(pred, y_expected, random_mask)
                total_test_loss += test_loss.detach().item()

                # Calculate accuracy.
                y_predicted = torch.argmax(pred, dim=-1)
                true_labels = y_expected[random_mask]
                predicted_labels = y_predicted[random_mask]
                correct_predictions += (true_labels == predicted_labels).sum().item()
                total_predictions += true_labels.size(0)
                predicted_labels_list += predicted_labels.tolist()
                true_labels_list += true_labels.tolist()
                accuracy_list.append((true_labels == predicted_labels).sum().item() / true_labels.size(0))

    # Compute average loss and accuracy.
    avg_test_loss = total_test_loss / (len(dataloader)*iterations)
    accuracy = correct_predictions / total_predictions

    # Log and return the results.
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return avg_test_loss, masked_tokens, accuracy, y_expected, y_predicted, random_mask, true_labels_list, predicted_labels_list, accuracy_list


def test_loop_topk(model, dataloader, src_vocab , iterations=1, use_fusion_embeddings=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    model.eval()
    total_test_loss = 0
    correct_predictions_top1 = 0
    correct_predictions_top5 = 0
    correct_predictions_top10 = 0
    total_predictions = 0

    # Additional lists for accuracy calculations
    accuracy_list_top1 = []
    accuracy_list_top5 = []
    accuracy_list_top10 = []

    with torch.no_grad():
        for _ in range(iterations):
            for X in dataloader:
                 # Move tensors to the same device as the model.
                for k in X.keys():
                    if k != 'user':
                        X[k] = X[k].to(device)

                # Extract features and masked tokens.
                features = [k for k in X.keys() if k != 'y' and k != 'user']
                masked_tokens, random_mask = build_dataset.torch_mask_tokens(X['y'], src_vocab)

                # Apply mask.
                for k in features:
                    X[k][random_mask] = 0 

                # Prepare inputs for the model.
                y_input = [masked_tokens, *[X[k] for k in features]]
                y_expected = X['y']

                # Create a mask for non-padded tokens.
                src_mask = ((X['y'] != 0).int()).to(device)
                src_mask = src_mask.unsqueeze(-1)
            
                # Get model predictions.
                if use_fusion_embeddings:
                    pred = model(y_input, src_mask)
                else:
                    pred = model(masked_tokens, src_mask) # masked_tokens if without FusionEmbeddings otherwise y_input
                
                test_loss = masked_loss(pred, y_expected, random_mask)
                total_test_loss += test_loss.detach().item()

                # Calculate top-1, top-5, and top-10 accuracy
                top1 = torch.argmax(pred, dim=-1)
                top5 = torch.topk(pred, 5, dim=-1).indices
                top10 = torch.topk(pred, 10, dim=-1).indices

                true_labels = y_expected[random_mask]  # Assuming this is how you get the true labels

                # Update top-1 accuracy calculations
                correct_predictions_top1 += (true_labels == top1[random_mask]).sum().item()

                # Update top-5 accuracy calculations
                correct_in_top5 = sum([true_labels[i] in top5[random_mask][i] for i in range(true_labels.size(0))])
                correct_predictions_top5 += correct_in_top5

                # Update top-10 accuracy calculations
                correct_in_top10 = sum([true_labels[i] in top10[random_mask][i] for i in range(true_labels.size(0))])
                correct_predictions_top10 += correct_in_top10

                total_predictions += true_labels.size(0)

    avg_test_loss = total_test_loss / (len(dataloader) * iterations)
    accuracy_top1 = correct_predictions_top1 / total_predictions
    accuracy_top5 = correct_predictions_top5 / total_predictions
    accuracy_top10 = correct_predictions_top10 / total_predictions

    print(f"Test Loss: {avg_test_loss:.4f}, Top-1 Accuracy: {accuracy_top1:.4f}, Top-5 Accuracy: {accuracy_top5:.4f}, Top-10 Accuracy: {accuracy_top10:.4f}")
    return avg_test_loss, accuracy_top1, accuracy_top5, accuracy_top10
