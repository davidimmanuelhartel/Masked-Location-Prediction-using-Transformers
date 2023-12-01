from torch.utils.data import Dataset
import torch
import pandas as pd
import ast
from collections import Counter

# Class that extends torch.utils.data.Dataset class
# contains functions connected to the creation of the dataset

class BertMobilityDataset(Dataset):
    def __init__(self, df, rank_dict, vocab= None):
        # assuming df has the columns 'user, 'user_week', 'date', 'pos'
        self.df = df  
        self.df['date'] = pd.to_datetime(self.df['date']).dt.date  
        self.df["user"] = self.df["user"].astype(int).astype(str)
        self.df['rank'] = self.df.apply(lambda row: rank_dict.get(row['user_week'], {}).get(row['pos'], 0), axis=1)
        self.users = df['user'].unique()
        self.user_weeks = df['user_week'].unique()  
        
        self.vocab = vocab if vocab is not None else self.build_vocab()  
        self.vocab_size = len(self.vocab)
        self.max_sequence_length = self.calculate_max_sequence_length()
        # self.highest_rank_pos_dict = self.get_highest_rank_pos_dict()
        # Convert string lists to actual lists if needed
        if isinstance(self.df['pos'].iloc[0], str):
            self.df['pos'] = self.df['pos'].apply(ast.literal_eval)  

    def build_vocab(self):
        # Add special tokens and user tokens
        special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]']
        user_tokens = ['[U_' + ''.join(filter(str.isdigit, str(user))) + ']' for user in self.users]
        # Build vocab from positions
        counter = Counter([str(pos) for pos in self.df['pos'].tolist()])
        vocab = sorted(counter, key=counter.get, reverse=True)
        return special_tokens + user_tokens + vocab

    
    # Function to calculate frequency of the vocabulary words
    def calculate_vocab_freq(self):
        counter = Counter([str(pos) for pos in self.df['pos'].tolist()])  
        # return the counter with frequencies as percentages, sorted by frequency
        return {k: round(v / sum(counter.values()),2) for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}

    # Function to calculate the maximum sequence length for a week
    def calculate_max_sequence_length(self):
        max_len = 0
        for name, group in self.df.groupby('user_week'):
            length = len(group) + len(group['date'].unique()) + 1  # +1 for [CLS] and + number_of_days for [SEP]
            if length > max_len:
                max_len = length
        return max_len
    
    # Function to calculate the average sequence length for a week
    def calculate_avg_sequence_length(self):
        total_len = 0
        for name, group in self.df.groupby('user_week'):
            total_len += len(group)
        return total_len / len(self.user_weeks)

    # Function to encode positions to their corresponding vocab indices
    def encode_positions(self, positions):  
        return [self.vocab.index(str(pos)) for pos in positions]
    
    # decode vocab indices to positions
    def decode_positions(self, indices):
        return [self.vocab[i] for i in indices]
    
    # Function find the top rank position for each user_week
    def get_highest_rank_pos_dict(self):
        highest_rank_positions = self.df.loc[self.df.groupby('user_week')['rank'].idxmin()]
        highest_rank_pos_dict = dict(zip(highest_rank_positions['user_week'], highest_rank_positions['pos']))
        return highest_rank_pos_dict

    # Return the total number of user_weeks
    def __len__(self):  
        return len(self.user_weeks)


    # Get a specific item from the dataset
    def __getitem__(self, index):
        user_week = self.user_weeks[index]
        user_data = self.df[self.df['user_week'] == user_week].sort_values(by=['date'])
        user = user_week.split('_')[0].lstrip('u')  # Remove 'u' prefix and get the user part
        user_token = '[U_' + user + ']'
        input_ids = [self.vocab.index(user_token)]  # Start with the user-specific token instead of CLS token
        # input_ids = [self.vocab.index('[CLS]')]

        ranks = [0]
        timestamps = [0]

        for _, row in user_data.iterrows():
            pos = row['pos']
            encoded_pos = self.encode_positions([pos])
            input_ids += encoded_pos

            # Append timestamp and rank information for each position
            timestamp = pd.to_datetime(row['date']).value // 1e9  # Convert to Unix timestamp
            timestamps.append(int(timestamp))  # Explicit conversion to integer
            ranks.append(row['rank'])


        padding_length = self.max_sequence_length - len(input_ids)
        input_ids += [self.vocab.index('[PAD]')] * padding_length
        timestamps += [0] * padding_length
        ranks += [0] * padding_length

        # Convert dates to a tensor 
        timestamp_tensors = torch.tensor(timestamps, dtype=torch.long)
        rank_tensors = torch.tensor(ranks, dtype=torch.long)

        dict_return = {
            'y': torch.tensor(input_ids, dtype=torch.long),
            #'date_time': timestamp_tensors,
            'rank': rank_tensors,
            'user_week': user_week   
        }
        return dict_return

def compute_rank_dict(df):
    # assuming df has columns "user" and "stoplocation"
    # count visits for each user-location pair
    visit_counts = df.groupby(['user', 'pos']).size().reset_index(name='visit_count')

    # Rank locations for each user based on visit counts
    visit_counts['rank'] = visit_counts.groupby('user')['visit_count'].rank(ascending=False, method='min')

    # Convert ranks to integers
    visit_counts['rank'] = visit_counts['rank'].astype(int)

    # Create a dictionary to store the ranks
    rank_dict = {}
    for user, group in visit_counts.groupby('user'):
        rank_dict[user] = dict(zip(group['pos'], group['rank']))

    return rank_dict


def torch_mask_tokens(inputs, n_tokens):
    """
        Prepare inputs/labels for 
        masked language modeling: 
        Mask 15% of all tokens
        80% MASK, 10% random, 10% original.
    """
    inputs = torch.clone(inputs)
    
    # Identify non-padding tokens - exclude 0, 1, 2
    non_padding_indices = (inputs != 0) & (inputs != 1) & (inputs != 2)
    d = non_padding_indices.sum().item()
    # Calculate the number of tokens to mask (15% of non-padding tokens)
    k = max(1, int(0.15 * d))
    rand_vec = torch.rand(d)
    k_th_quant = torch.topk(rand_vec, k, largest=False)[0][-1]
    
    # Create a mask for tokens to be masked
    mask_for_non_padding = rand_vec <= k_th_quant
    mask = torch.zeros_like(inputs).bool()
    mask[non_padding_indices] = mask_for_non_padding
    
    # Replace 80% of masked tokens with [MASK]
    indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & mask
    inputs[indices_replaced] = 3  # [MASK] token is 3
    
    # Randomly replace half of the remaining 20% tokens
    indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & mask & ~indices_replaced
    random_words = torch.randint(4, n_tokens, inputs.shape, dtype=torch.long, device=inputs.device)
    inputs[indices_random] = random_words[indices_random]
    
    # test:remove indices random from mask
    mask[indices_random] = 0
    
    return inputs, mask

def create_user_week(row):
    # Convert the 'user' value to an integer and then to a string
    user_str = str(int(float(row['user'])))

    # Extract the week number from the 'date' column
    date = pd.to_datetime(row['date'])
    week_int = date.isocalendar()[1]  # Get the ISO week number
    week_str = f"{week_int:02d}"  # Keep zero-padding for week number

    # Combine them into a single string with '_w' in between
    return f"u{user_str}_w{week_str}"


import pandas as pd
import numpy as np
from tqdm import tqdm

def stratified_user_week_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, ensure_user_in_all_sets=True):
    """
    Splits the DataFrame into train, validation, and test sets such that each 'user_week' group is kept intact,
    and 'user' is as evenly distributed as possible across the sets. Optionally ensures each user is present in all sets.

    Parameters:
    df (DataFrame): The DataFrame to split.
    train_ratio (float): The proportion of the dataset to include in the train split.
    val_ratio (float): The proportion of the dataset to include in the validation split.
    test_ratio (float): The proportion of the dataset to include in the test split.
    ensure_user_in_all_sets (bool): If True, ensures that each user is represented in all sets.

    Returns:
    DataFrame, DataFrame, DataFrame: Train, validation, and test DataFrames.
    """

    # Group data by 'user_week'
    grouped = df.groupby('user_week')
    user_week_indices = {user_week: group.index.tolist() for user_week, group in grouped}

    train_indices, val_indices, test_indices = [], [], []

    # Process each user
    for user in df['user'].unique():
        user_data = df[df['user'] == user]
        user_weeks = user_data['user_week'].unique()
        np.random.shuffle(user_weeks)

        # Ensure each user is in all sets if required
        if ensure_user_in_all_sets and len(user_weeks) >= 3:
            train_indices.extend(user_week_indices[user_weeks[0]])
            val_indices.extend(user_week_indices[user_weeks[1]])
            test_indices.extend(user_week_indices[user_weeks[2]])
            user_weeks = user_weeks[3:]

        for group in user_weeks:
            total_assigned = len(train_indices) + len(val_indices) + len(test_indices)
            current_train_ratio = len(train_indices) / total_assigned if total_assigned > 0 else 0
            current_val_ratio = len(val_indices) / total_assigned if total_assigned > 0 else 0

            # Assign to the dataset with the lowest current ratio compared to the target ratio
            if current_train_ratio < train_ratio:
                train_indices.extend(user_week_indices[group])
            elif current_val_ratio < val_ratio:
                val_indices.extend(user_week_indices[group])
            else:
                test_indices.extend(user_week_indices[group])

    # Create final datasets
    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    return train_df, val_df, test_df
