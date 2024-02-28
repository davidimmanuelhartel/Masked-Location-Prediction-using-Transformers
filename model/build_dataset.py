from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import ast
import math
from collections import Counter

# Class that extends torch.utils.data.Dataset class
# contains functions connected to the creation of the dataset

class BertMobilityDataset(Dataset):
    def __init__(self, df, rank_dict, vocab= None, user_embedding = False):
        # assuming df has the columns 'user, 'user_week', 'date', 'pos'
        
        # prepare data frame for further processing
        self.df = df  
        self.df['date'] = pd.to_datetime(self.df['date'])  
        self.df["user"] = self.df["user"].astype(int).astype(str)
        self.user_embedding = user_embedding
        
        # assign ranks based on rank_dict
        self.df['rank'] = self.df.apply(lambda row: rank_dict.get(row['user'], {}).get(row['pos'], 0), axis=1) 
        
        # init variables
        self.users = df['user'].unique()
        self.user_weeks = df['user_week'].unique()  
        self.vocab = vocab if vocab is not None else self.build_vocab()  
        self.vocab_size = len(self.vocab)
        self.max_sequence_length = self.calculate_max_sequence_length()
        
        # Convert string lists to actual lists if needed
        if isinstance(self.df['pos'].iloc[0], str):
            self.df['pos'] = self.df['pos'].apply(ast.literal_eval) 
        
    def build_vocab(self):
        # Add special tokens and user tokens
        special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]']
        if self.user_embedding:
            user_tokens = ['[U_' + ''.join(filter(str.isdigit, str(user))) + ']' for user in self.users]
        # Build vocab from positions
        counter = Counter([str(pos) for pos in self.df['pos'].tolist()])
        vocab = sorted(counter, key=counter.get, reverse=True)
        if self.user_embedding:
            return special_tokens + user_tokens + vocab
        else:
            return special_tokens + vocab

    
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
    
    # Function find the top rank position for each user
    def get_highest_rank_pos_dict(self):
        highest_rank_positions = self.df.loc[self.df.groupby('user')['rank'].idxmin()]
        highest_rank_pos_dict = dict(zip(highest_rank_positions['user'], highest_rank_positions['pos']))
        return highest_rank_pos_dict
    
    # Calculate the probability of each location for each user based on its frequency in the dataset
    def calculate_location_probabilities(self):
            user_location_probabilities = {}
            # Group by user and calculate probabilities for each location
            for user, group in self.df.groupby('user'):
                location_counts = Counter(group['pos'])
                total_locations = sum(location_counts.values())
                location_probabilities = {loc: count / total_locations for loc, count in location_counts.items()}
                user_location_probabilities[user] = location_probabilities
            return user_location_probabilities
    # Calculate the transition probabilities for each user's locations.

    def calculate_transition_probabilities(self):
            user_transition_probs = {}
            for user, group in self.df.groupby('user'):
                # Sort the group by date
                sorted_group = group.sort_values(by='date')

                # Initialize a transition matrix for each user
                transitions = {loc: Counter() for loc in self.vocab[4:]}  # Excluding special tokens
                # Previous position placeholder
                prev_pos = None

                # Iterate through each position for the user
                for _, row in sorted_group.iterrows():
                    current_pos = str(row['pos'])
                    if prev_pos is not None and current_pos != prev_pos: #exclude consecutive duplicates (e.g. between user weeks)
                        # Count the transition from the previous position to the current position
                        transitions[prev_pos][current_pos] += 1
                    prev_pos = current_pos

                # Convert counts to probabilities
                for loc in transitions:
                    total_transitions = sum(transitions[loc].values())
                    if total_transitions > 0:
                        transitions[loc] = {k: v / total_transitions for k, v in transitions[loc].items()}

                user_transition_probs[user] = transitions

            return user_transition_probs
    
    def calculate_periodic_feature(self, value, max_value):
        if value == 0:  # Padding value
            return 0, 0
        # Scale the value to a 0 to 2π interval
        x = (value / max_value) * 2 * math.pi
        time_x = 0.5 * (math.sin(x) + 1)
        time_y = 0.5 * (math.cos(x) + 1)
        return time_x, time_y


    # Return the total number of user_weeks
    def __len__(self):  
        return len(self.user_weeks)


    # Get a specific item from the dataset
    def __getitem__(self, index):
        user_week = self.user_weeks[index]
        user_data = self.df[self.df['user_week'] == user_week].sort_values(by=['date'])
        # Get the user part from user_week and remove the 'u' prefix 
        user = str(int(user_week.split('_')[0].lstrip('u')))  # Remove 'u' prefix and get the user part # int(user_week.split('_')[0].lstrip('u')) if without usertoken
        if self.user_embedding: 
            user_token = '[U_' + user + ']'
            input_ids = [self.vocab.index(user_token)]  # Start with the user-specific token instead of CLS token
        else:
            input_ids = [self.vocab.index('[CLS]')] # Start with the CLS token
        ranks = [0] # [0] for the CLS/user-specific token
        day_of_week = [0]  # list to store day of week
        hour_of_day = [0]  # list to store time of the day

        for _, row in user_data.iterrows():
            pos = row['pos']
            encoded_pos = self.encode_positions([pos])
            input_ids += encoded_pos

            # Append features  for each position
            ranks.append(row['rank'])
            day_of_week.append(row['date'].weekday()+1) # Monday is 1 and Sunday is 7
            hour_of_day.append(row['date'].hour +1 ) # 1-24
            
        

        padding_length = self.max_sequence_length - len(input_ids)
        input_ids += [self.vocab.index('[PAD]')] * padding_length
        ranks += [0] * padding_length
        day_of_week += [0] * padding_length
        hour_of_day += [0] * padding_length
        
        # Calculate periodic time features independently
        time_xs_day, time_ys_day = zip(*[self.calculate_periodic_feature(day, 7) for day in day_of_week])  # For day of the week
        time_xs_hour, time_ys_hour = zip(*[self.calculate_periodic_feature(hour, 24) for hour in hour_of_day])  # For time of the day

        # Convert features to tensors
        rank_tensors = torch.tensor(ranks, dtype=torch.long)
        day_of_week_tensors = torch.tensor(day_of_week, dtype=torch.long)
        hour_of_day_tensors = torch.tensor(hour_of_day, dtype=torch.long)
        time_x_day_tensors = torch.tensor(time_xs_day, dtype=torch.float)
        time_y_day_tensors = torch.tensor(time_ys_day, dtype=torch.float)
        time_x_hour_tensors = torch.tensor(time_xs_hour, dtype=torch.float)
        time_y_hour_tensors = torch.tensor(time_ys_hour, dtype=torch.float)

        dict_return = {
            'y': torch.tensor(input_ids, dtype=torch.long),
            'rank': rank_tensors,
            'user': user,
            # 'day_of_week': day_of_week_tensors,
            # 'time_of_the_day': hour_of_day_tensors,
            # 'time_x_day': time_x_day_tensors,
            # 'time_y_day': time_y_day_tensors,
            # 'time_x_hour': time_x_hour_tensors,
            # 'time_y_hour': time_y_hour_tensors,
        }
        return dict_return

def compute_rank_dict(df):
    # count visits for each user-location pair
    visit_counts = df.groupby(['user', 'pos']).size().reset_index(name='visit_count')

    # Rank locations for each user based on visit counts
    visit_counts['rank'] = visit_counts.groupby('user')['visit_count'].rank(ascending=False, method='min')

    # Convert ranks to integers
    visit_counts['rank'] = visit_counts['rank'].astype(int)

    # Create a dictionary to store the ranks
    rank_dict = {}
    for user, group in visit_counts.groupby('user'):
        user_no = str(int(float(user)))
        rank_dict[user_no] = dict(zip(group['pos'], group['rank']))
    return rank_dict


def torch_mask_tokens(inputs, n_tokens, seed=0):
    """
        Prepare inputs/labels for 
        masked language modeling: 
        Mask 15% of all tokens
        80% MASK, 10% random, 10% original.
    """
    inputs = torch.clone(inputs)
    # torch.manual_seed(seed)
    
    # Identify non-padding tokens - exclude 0, 1, 2
    non_padding_indices = (inputs != 0) & (inputs != 1) & (inputs != 2)
    non_padding_indices[:, 0] = False  # Exclude the first token of each sequence
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



def stratified_user_week_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, ensure_user_in_all_sets=True,random_seed=None):
    """
    Splits the DataFrame into train, validation, and test sets such that each 'user_week' group is kept intact,
    and 'user' is as evenly distributed as possible across the sets. Optionally ensures each user is represented in all sets.

    Parameters:
    df (DataFrame): The DataFrame to split.
    train_ratio (float): The proportion of the dataset to include in the train split.
    val_ratio (float): The proportion of the dataset to include in the validation split.
    test_ratio (float): The proportion of the dataset to include in the test split.
    ensure_user_in_all_sets (bool): If True, ensures that each user is represented in all sets.

    Returns:
    DataFrame, DataFrame, DataFrame: Train, validation, and test DataFrames.
    """

    # Check if the ratios sum to 1
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.")

    # Set a random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Group data by 'user_week'
    grouped = df.groupby('user_week')
    user_week_indices = {user_week: group.index.tolist() for user_week, group in grouped}

    train_indices, val_indices, test_indices = [], [], []

    # Process each user
    for user in df['user'].unique():
        user_data = df[df['user'] == user]
        user_weeks = user_data['user_week'].unique()

        # Discard users with less than 3 'user_weeks'
        if len(user_weeks) < 3:
            continue

        np.random.shuffle(user_weeks)

        # Ensure each user is in all sets if required
        if ensure_user_in_all_sets:
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

