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
            



def fit(model, baseline_model, opt, train_dataloader, val_dataloader, epochs, src_vocab):
    """
    This method performs the training and validation of the model over multiple epochs.
    Inspired by "A detailed guide to Pytorch's nn.Transformer() module," by Daniel Melchor.
    https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    # Lists to store training and validation loss for each epoch
    train_loss_list= []
    train_accuracy_list = []
    validation_loss_list = []
    baseline_loss_list = []
    val_accuracy_list = []
    val_baseline_accuracy_list = []
    
    
    # Start training and validation
    print("Training and validating model")
    for epoch in range(epochs):
        
        # Print the epoch number for better logging
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)
        
        # Perform one epoch of training and get the training loss
        train_loss, total_train_acc = train_loop(model, opt, train_dataloader, src_vocab)
        
        # Append this epoch's training loss to the list
        train_loss_list += [train_loss]
        
        # Perform one epoch of validation and get the validation loss
        validation_loss, baseline_loss, total_val_acc, total_baseline_acc  = validation_loop(model, baseline_model, val_dataloader, src_vocab)
        
        # Append this epoch's validation loss to the list
        validation_loss_list += [validation_loss]
        baseline_loss_list += [baseline_loss]
        val_accuracy_list += [total_val_acc]
        val_baseline_accuracy_list += [total_baseline_acc]
        train_accuracy_list += [total_train_acc]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Baseline loss: {baseline_loss:.4f}")
        print()
        
    return train_loss_list, validation_loss_list, baseline_loss_list, train_accuracy_list, val_accuracy_list, val_baseline_accuracy_list


def train_loop(model, opt, dataloader, src_vocab):
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
        masked_tokens, random_mask = torch_mask_tokens(X['y'], src_vocab)
        
        # Mask the features with the random mask.
        for k in features:
            X[k][random_mask] = 0 
        
        # Prepare input and expected output for the model.
        y_input = [masked_tokens, *[X[k].view(X[k].size(0), X[k].size(1), -1).float() for k in features]]
        y_expected = X['y']
        
        # Create a mask to ignore padding in the sequence.
        src_mask = ((X['y'] != 0).int()).to(device)
        src_mask = src_mask.unsqueeze(-1)
        
        # Get model predictions.
        y_predicted = model(y_input, src_mask) # y_input if using with FusionEmbeddings. otherwise masked_masked tokens

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
    print('Accuracy train:', total_train_acc.item())

    
    # Return the average loss for this epoch.
    return total_loss / len(dataloader), total_train_acc


def validation_loop(model,baseline_model, dataloader, src_vocab):
    """
    This method performs one epoch of validation on the model.
    It was inspired by "A detailed guide to Pytorch's nn.Transformer() module.",
    by Daniel Melchor.
    """
    
    # Set the model to evaluation mode.
    model.eval()
    
    # Initialize variables to keep track of total loss and number of batches.
    total_val_loss, total_baseline_loss = 0,0 
    accuracy_list, baseline_accuracy_list = [], []
    total_val_acc, total_baseline_acc = 0,0
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
            masked_tokens, random_mask = torch_mask_tokens(X['y'], src_vocab)

            # Apply the random mask to all the features.
            for k in features:
                X[k][random_mask] = 0 

            # Prepare the input for the model. It includes masked tokens and features.
            y_input = [masked_tokens, *[X[k].view(X[k].size(0), X[k].size(1), -1).float() for k in features]]
            y_expected = X['y']

            # Create a mask to ignore padded tokens.
            src_mask = ((X['y'] != 0).int()).to(device)
            src_mask = src_mask.unsqueeze(-1)

            # Get the model's predictions.
            y_predicted = model(y_input, src_mask) # masked_tokens if without FusionEmbeddings

            # Get the baseline predictions
            baseline_pred = baseline_model.predict(masked_tokens, y_expected, X['user'])

            # Calculate the loss value.
            val_loss = masked_loss(y_predicted, y_expected, random_mask)
            baseline_loss = masked_loss(baseline_pred, y_expected, random_mask)
            
            # Add the batch's loss to the total loss for this epoch.
            total_val_loss += val_loss.detach().item()
            total_baseline_loss += baseline_loss.detach().item()
            
            # Increase the batch counter.
            nn += 1
        
            # Calculate accuracy for the current batch.
            predicted_labels = torch.argmax(y_predicted, dim=-1)
            true = y_expected[random_mask]
            mdl = predicted_labels[random_mask]
            acc = sum(true == mdl) / true.size()[0] if true.size()[0] > 0 else 0
            accuracy_list.append(acc)
            
            # Calculate the accuracy for the baseline model.
            baseline_predicted_labels = torch.argmax(baseline_pred, dim=-1)
            baseline_mdl = baseline_predicted_labels[random_mask]
            baseline_acc = sum(true == baseline_mdl) / true.size()[0]
            baseline_accuracy_list.append(baseline_acc)
    
    # Calculate and print the average accuracy for this epoch.
    total_val_acc = sum(accuracy_list) / len(accuracy_list)
    total_baseline_acc = sum(baseline_accuracy_list) / len(baseline_accuracy_list)
    print('Accuracy val:', total_val_acc.item())
    
    # Return the average loss over all batches.
    return total_val_loss / len(dataloader), total_baseline_loss / len(dataloader), total_val_acc, total_baseline_acc


def masked_loss(y_predicted, 
                y_expected, 
                mask):
    
    y_predicted = y_predicted[mask]
    y_expected = y_expected[mask]
    
    loss_fn = F.cross_entropy(y_predicted, y_expected)

    return loss_fn