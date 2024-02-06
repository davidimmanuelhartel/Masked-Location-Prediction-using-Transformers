import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, Tensor
from torch.utils.data import DataLoader


# local imports
from build_dataset import torch_mask_tokens


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# implementation of the "Noam" learning rate scheduler, commonly used with Transformer models
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Updates parameters and rate and perform a gradient update"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "calculates the learning rate based on the current step"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
            


def fit(model, opt, train_dataloader, val_dataloader, epochs, ntokens, baseline_model_simple, baseline_model_sampled, baseline_model_markov_1, use_fusion_embeddings=False):
    """
    This method performs the training and validation of the model over multiple epochs.
    If a baseline model is provided, it will also be validated for comparison.
    Inspired by "A detailed guide to Pytorch's nn.Transformer() module," by Daniel Melchor.
    https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    # Lists to store training and validation loss for each epoch
    train_loss_list = []
    train_accuracy_list = []
    validation_loss_list = []
    val_accuracy_list = []

    # Lists for baseline model, if provided
    val_baseline_simple_accuracy_list = []
    val_baseline_sampled_accuracy_list = []
    val_baseline_markov_1_accuracy_list = []

    # Start training and validation
    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        # Perform one epoch of training and get the training loss and accuracy
        train_loss, total_train_acc = train_loop(model, opt, train_dataloader, ntokens, use_fusion_embeddings)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(total_train_acc)

        # Perform one epoch of validation and get the validation loss and accuracy
        validation_loss, total_val_acc = validation_loop(model, val_dataloader, ntokens, use_fusion_embeddings)
        validation_loss_list.append(validation_loss)
        val_accuracy_list.append(total_val_acc)

        print(f"Training loss: {train_loss:.4f}, Training Accuracy: {total_train_acc:.4f}")
        print(f"Validation loss: {validation_loss:.4f}, Validation Accuracy: {total_val_acc:.4f}")

        # Validate the baseline model if provided
        total_baseline_simple_acc, total_baseline_sampled_acc, total_baseline_markov_1_acc = baseline_validation_loop(baseline_model_simple, baseline_model_sampled, baseline_model_markov_1, val_dataloader, ntokens)
        val_baseline_simple_accuracy_list.append(total_baseline_simple_acc)
        val_baseline_sampled_accuracy_list.append(total_baseline_sampled_acc)
        val_baseline_markov_1_accuracy_list.append(total_baseline_markov_1_acc)
        print(f"Baseline Simple Accuracy: {total_baseline_simple_acc:.4f}")
        print(f"Baseline Sampled Accuracy: {total_baseline_sampled_acc:.4f}")
        print(f"Baseline Markov-1 Accuracy: {total_baseline_markov_1_acc:.4f}")
        print()

    return (train_loss_list, train_accuracy_list, validation_loss_list, val_accuracy_list,
            val_baseline_simple_accuracy_list,val_baseline_sampled_accuracy_list,val_baseline_markov_1_accuracy_list,)



def train_loop(model, opt, dataloader, ntokens, use_fusion_embeddings=False):
    """
    This method performs one epoch of training on the model.
    It iterates through the training data, performs forward and backward passes, and updates the model parameters. 
    It also calculates the training loss and accuracy
    It was inspired by "A detailed guide to Pytorch's nn.Transformer() module.",
    by Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    accuracy_list = []
    total_train_acc = 0

    # Set the model to training mode.
    model.train()
    
    # Initialize total loss and batch counter.
    total_loss = 0
    nn = 0
    
    # Loop through each batch in the dataloader.
    for X in dataloader:
        print("training batch:", nn, "of", len(dataloader), end="\r")
        
        # Move all tensors in the current batch to the specified device (CPU or GPU).
        for k in X.keys():
            if k != 'user':
                X[k] = X[k].to(device)
        
        # Extract feature names (keys that are not the target 'y').
        features = [k for k in X.keys() if k != 'y' and k != 'user']

        # Create masked tokens and random masks for the target sequence.
        masked_tokens, random_mask = torch_mask_tokens(X['y'], ntokens)
        # Mask the features with the random mask.
        
        for k in features:
            X[k][random_mask] = 0
            
        # Prepare input and expected output for the model.
        y_input = [masked_tokens, *[X[k] for k in features]]
        y_expected = X['y']
        
        # Create a mask to ignore padding in the sequence.
        src_mask = ((X['y'] != 0).int()).to(device)
        src_mask = src_mask.unsqueeze(-1)
     
        # Get model predictions.
        if use_fusion_embeddings:
            y_predicted = model(y_input, src_mask)
        else:
            y_predicted = model(masked_tokens, src_mask) # y_input if using with FusionEmbeddings. otherwise masked_tokens

        # Calculate loss using masked tokens.
        loss = masked_loss(y_predicted, y_expected, random_mask)
        
        # Zero the gradients before backpropagation.
        opt.zero_grad()
        
        # Backpropagate the loss.
        loss.backward()
        
        # Update model parameters.
        opt.step()
        
        # Accumulate the total loss.
        total_loss += loss.detach().item()
        
        # Increment the batch counter.
        nn += 1
        
        # Calculate accuracy for the current batch.
        predicted_labels = torch.argmax(y_predicted, dim=-1)
        true = y_expected[random_mask]
        mdl = predicted_labels[random_mask]
        acc = sum(true == mdl) / true.size()[0] if true.size()[0] > 0 else 0
        accuracy_list.append(acc)
         
    
    total_train_acc = sum(accuracy_list) / len(accuracy_list)

    
    # Return the average loss for this epoch.
    return total_loss / len(dataloader), total_train_acc


def validation_loop(model, dataloader, ntokens, use_fusion_embeddings=False):
    """
    This method performs one epoch of validation on the model.
    It was inspired by "A detailed guide to Pytorch's nn.Transformer() module.",
    by Daniel Melchor.
    """
    
    # Set the model to evaluation mode.
    model.eval()
    
    # Initialize variables to keep track of total loss and number of batches.
    total_val_loss = 0
    accuracy_list = []
    total_val_acc = 0
    nn = 0
    
    # Use torch.no_grad() to deactivate the autograd engine and reduce memory usage and speed up computations.
    with torch.no_grad():
        # Iterate over each batch from the DataLoader.
        for X in dataloader:
            print("validation batch:", nn, "of", len(dataloader), end="\r")
            
            # Move all tensors to the same device as the model.
            for k in X.keys():
                if k != 'user':
                    X[k] = X[k].to(device)
            # Extract all features from the batch (every key in the dictionary that is not 'y').
            features = [k for k in X.keys() if k != 'y' and k != 'user']

            # Create masked tokens and random mask.
            masked_tokens, random_mask = torch_mask_tokens(X['y'], ntokens)

            # Apply the random mask to all the features.
            for k in features:
                X[k][random_mask] = 0 

            # Prepare the input for the model. It includes masked tokens and features.
            y_input = [masked_tokens, *[X[k] for k in features]]
            y_expected = X['y']

            # Create a mask to ignore padded tokens.
            src_mask = ((X['y'] != 0).int()).to(device)
            src_mask = src_mask.unsqueeze(-1)

            # Get model predictions.
            if use_fusion_embeddings:
                y_predicted = model(y_input, src_mask)
            else:
                y_predicted = model(masked_tokens, src_mask) # y_input if using with FusionEmbeddings. otherwise masked_tokens

            # Calculate the loss value.
            val_loss = masked_loss(y_predicted, y_expected, random_mask)
            
            # Add the batch's loss to the total loss for this epoch.
            total_val_loss += val_loss.detach().item()
            
            # Increase the batch counter.
            nn += 1
        
            # Calculate accuracy for the current batch.
            predicted_labels = torch.argmax(y_predicted, dim=-1)
            true = y_expected[random_mask]
            mdl = predicted_labels[random_mask]
            acc = sum(true == mdl) / true.size()[0] if true.size()[0] > 0 else 0
            accuracy_list.append(acc)
    
    # Calculate and print the average accuracy for this epoch.
    total_val_acc = sum(accuracy_list) / len(accuracy_list)
    
    # Return the average loss over all batches.
    return total_val_loss / len(dataloader), total_val_acc



def baseline_validation_loop(baseline_model_simple, baseline_model_sampled, baseline_model_markov_1, dataloader, ntokens):
    """
    This method performs one epoch of validation on the baseline model.
    """
    
    # Initialize variables to keep track of total loss and number of batches.
    baseline_simple_acc_list = []
    baseline_sampled_acc_list = []
    baseline_markov_1_acc_list = []
    total_baseline_simple_acc = 0
    total_baseline_sampled_acc = 0
    total_baseline_markov_1_acc = 0
    nn = 0
    
    # Use torch.no_grad() to deactivate the autograd engine and reduce memory usage and speed up computations.
    with torch.no_grad():
        # Iterate over each batch from the DataLoader.
        for X in dataloader:
            
            # Move all tensors to the same device as the model.
            for k in X.keys():
                if k != 'user':
                    X[k] = X[k].to(device)
                    
            # Extract all features from the batch (every key in the dictionary that is not 'y').
            features = [k for k in X.keys() if k != 'y' and k != 'user']

            # Create masked tokens and random mask.
            masked_tokens, random_mask = torch_mask_tokens(X['y'], ntokens)

            # Apply the random mask to all the features.
            for k in features:
                X[k][random_mask] = 0 

            # Prepare the input for the model. It includes masked tokens and features.
            y_expected = X['y']

            # Get the baseline predictions
            predicted_labels_simple = baseline_model_simple.predict(masked_tokens, random_mask, X['user'])
            predicted_labels_sampled = baseline_model_sampled.predict(masked_tokens, random_mask, X['user'])
            predicted_labels_markov_1 = baseline_model_markov_1.predict(masked_tokens, random_mask, X['user'])
            # Increase the batch counter.
            nn += 1
        
            # Calculate the accuracy for the baseline models 
            true = y_expected[random_mask]
            baseline_masked_labels_simple = predicted_labels_simple[random_mask]
            baseline_masked_labels_sampled = predicted_labels_sampled[random_mask]
            baseline_masked_labels_markov_1 = predicted_labels_markov_1[random_mask]
            baseline_simple_acc = sum(true == baseline_masked_labels_simple) / true.size()[0]
            baseline_sampled_acc = sum(true == baseline_masked_labels_sampled) / true.size()[0]
            baseline_markov_1_acc = sum(true == baseline_masked_labels_markov_1) / true.size()[0]
            baseline_simple_acc_list.append(baseline_simple_acc)
            baseline_sampled_acc_list.append(baseline_sampled_acc)
            baseline_markov_1_acc_list.append(baseline_markov_1_acc)
            
    # Calculate and print the average accuracy for this epoch.
    total_baseline_simple_acc = sum(baseline_simple_acc_list) / len(baseline_simple_acc_list)
    total_baseline_sampled_acc = sum(baseline_sampled_acc_list) / len(baseline_sampled_acc_list)
    total_baseline_markov_1_acc = sum(baseline_markov_1_acc_list) / len(baseline_markov_1_acc_list)
    
    # Return the average loss over all batches.
    return  total_baseline_simple_acc, total_baseline_sampled_acc, total_baseline_markov_1_acc


def masked_loss(y_predicted, 
                y_expected, 
                mask):
    
    y_predicted = y_predicted[mask]
    y_expected = y_expected[mask]
    
    loss_fn = F.cross_entropy(y_predicted, y_expected)

    return loss_fn