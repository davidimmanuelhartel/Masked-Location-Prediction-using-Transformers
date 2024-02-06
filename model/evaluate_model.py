import json
import torch
from datetime import datetime

def tensor_to_list(tensor):
    """
    Converts a PyTorch Tensor to a list.
    """
    return tensor.detach().cpu().tolist()

def convert_to_serializable(data):
    """
    Recursively converts elements in a data structure (dicts, lists, tuples) that are Tensors into lists.
    """
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [convert_to_serializable(item) for item in data]
    elif torch.is_tensor(data):
        return tensor_to_list(data)
    return data

def save_model_data( measures, config, variant):
    """
    Saves model parameters, training measures, and configuration to files with unique names.

    Parameters:
    - measures (tuple): Tuple containing training measures like loss and accuracy lists.
    - config (dict): Dictionary containing model configuration parameters.
    - variant (str): User-defined string to ensure uniqueness of file names.
    """

    # Construct file names with variant and date_time
    measures_filename = f'../../output/training_measures_{variant}.json'
    config_filename = f'../../output/model_config_{variant}.json'

    # Convert Tensors in measures to lists and create a dictionary
    # train_loss_list, train_accuracy_list, validation_loss_list, val_accuracy_list, val_baseline_simple_accuracy_list, val_baseline_sampled_accuracy_list, val_baseline_markov_1_accuracy_list
    
    measures_keys = ['train_loss_list', 'train_accuracy_list', 'validation_loss_list', 'val_accuracy_list', 'val_baseline_simple_accuracy_list', 'val_baseline_sampled_accuracy_list', 'val_baseline_markov_1_accuracy_list']
    measures_converted = {key: convert_to_serializable(value) 
                          for key, value in zip(measures_keys, measures)}

    # Save the training measures
    with open(measures_filename, 'w') as file:
        json.dump(measures_converted, file)

    # Save model configuration
    with open(config_filename, 'w') as file:
        json.dump(config, file)

    print(f"Saved measures to {measures_filename}, and config to {config_filename}")
    
def save_test_results(measures, config, variant):
    """
    Saves model parameters, training measures, and configuration to files with unique names.

    Parameters:
    - measures (tuple): Tuple containing training measures like loss and accuracy lists.
    - config (dict): Dictionary containing model configuration parameters.
    - variant (str): User-defined string to ensure uniqueness of file names.
    """

    # Construct file names with variant and date_time
    measures_filename = f'../../output/test_measures_{variant}.json'
    config_filename = f'../../output/test_config_{variant}.json'

    # Convert Tensors in measures to lists and create a dictionary
    # train_loss_list, train_accuracy_list, validation_loss_list, val_accuracy_list, val_baseline_simple_accuracy_list, val_baseline_sampled_accuracy_list, val_baseline_markov_1_accuracy_list
    
    measures_keys = ['avg_test_loss', 'masked_tokens', 'accuracy', 'y_expected', 'y_predicted', 'true_labels_list', 'predicted_labels_list', 'accuracy_list']
    measures_converted = {key: convert_to_serializable(value) 
                          for key, value in zip(measures_keys, measures)}

    # Save the training measures
    with open(measures_filename, 'w') as file:
        json.dump(measures_converted, file)

    # Save model configuration
    with open(config_filename, 'w') as file:
        json.dump(config, file)

    print(f"Saved measures to {measures_filename}, and config to {config_filename}")
    
import os 
def load_model_data(variant):
    """
    Loads model parameters, training measures, configuration, and dataset info from files.

    Parameters:
    - variant (str): The variant string used in the filenames.

    Returns:
    A tuple containing the training measures and configuration.
    """
    # print current path
    print(os.getcwd())
    # Construct file names with variant
    measures_filename = f'../../output/training_measures_{variant}.json'
    config_filename = f'../../output/model_config_{variant}.json'


    # Load the training measures
    with open(measures_filename, 'r') as file:
        measures = json.load(file)

    # Load model configuration
    with open(config_filename, 'r') as file:
        config = json.load(file)



    return  measures, config



