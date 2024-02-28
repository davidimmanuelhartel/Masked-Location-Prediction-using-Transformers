"""
Entropy Calculation Functions

This script contains functions for calculating various entropy measures based on location data.
These functions include:
- calculate_random_entropy: Calculate Random Entropy for each user in the DataFrame.
- calculate_temporal_uncorrelated_entropy: Calculate Temporal-Uncorrelated Entropy for each user in the DataFrame.
- calculate_actual_entropy_simplified: Calculate a simplified version of Actual Entropy for each user in the DataFrame.
- calculate_actual_entropy_advanced: Calculate the Actual Entropy as defined in the paper for each user in the DataFrame.
- calculate_lempel_ziv_entropy: Calculate the Lempel-Ziv entropy for each user in the DataFrame.
- calculate_predictability: Calculate the predictability based on the number of locations (N) and entropy (S_i).
"""

from itertools import combinations
import pandas as pd
import numpy as np
import math
from collections import Counter

def calculate_random_entropy(df):
    """
    Calculate Random Entropy for each user in the DataFrame.
    :param df: Pandas DataFrame with columns 'user' and 'pos'.
    :return: Dictionary of Random Entropy for each user.
    """
    return df.groupby('user')['pos'].nunique().apply(math.log2).to_dict()

def calculate_temporal_uncorrelated_entropy(df):
    """
    Calculate Temporal-Uncorrelated Entropy for each user in the DataFrame.
    :param df: Pandas DataFrame with columns 'user' and 'pos'.
    :return: Dictionary of Temporal-Uncorrelated Entropy for each user.
    """
    entropy_dict = {}
    for user, group in df.groupby('user'):
        location_counter = Counter(group['pos'])
        total_visits = sum(location_counter.values())
        entropy = 0
        for count in location_counter.values():
            probability = count / total_visits
            entropy -= probability * math.log2(probability)
        entropy_dict[user] = entropy
    return entropy_dict


def calculate_actual_entropy_simplified(df):
    """
    Calculate a simplified version of Actual Entropy for each user in the DataFrame.
    The function only considers the probability of the next location given the current location.
    :param df: Pandas DataFrame with columns 'user' and 'pos'.
    :return: Dictionary of Actual Entropy for each user.
    """
    entropy_dict = {}
    for user, group in df.groupby('user'):
        locations = group['pos'].tolist()
        location_sequence_counter = Counter([tuple(locations[i:i+2]) for i in range(len(locations)-1)])
        total_sequences = sum(location_sequence_counter.values())
        entropy = 0
        for count in location_sequence_counter.values():
            probability = count / total_sequences
            entropy -= probability * math.log2(probability)
        entropy_dict[user] = entropy
    return entropy_dict


def calculate_actual_entropy_advanced(df): # exponetial time complexity --> not feasible
    """
    Calculate the Actual Entropy as defined in the paper for each user in the DataFrame.
    :param df: Pandas DataFrame with columns 'user' and 'pos' representing the trajectory.
    :return: Dictionary of Actual Entropy for each user.
    """
    entropy_dict = {}

    for user, group in df.groupby('user'):
        trajectory = group['pos'].tolist()
        subsequence_counter = Counter()
        print("user", user)

        # Counting all possible subsequences
        print("counting all possible subsequences")
        for length in range(1, len(trajectory) + 1):
            print("length", length, "out of", len(trajectory) + 1, end='\r')
            for subsequence in combinations(trajectory, length):
                subsequence_counter[subsequence] += 1

        total_subsequences = sum(subsequence_counter.values())
        entropy = 0
        print("calculating entropy")
        for subsequence, count in subsequence_counter.items():
            probability = count / total_subsequences
            entropy -= probability * math.log2(probability)

        entropy_dict[user] = entropy

    return entropy_dict



import math

def calculate_lempel_ziv_entropy(df):
    """
    Calculate the Lempel-Ziv entropy for each user in the DataFrame.
    :param dataframe: Pandas DataFrame with columns 'user' and 'pos'.
    :return: Pandas Series with entropy for each user.
    """


    def lempel_ziv_entropy(location_series):
        """
        Calculate the entropy of a location history series using Lempel-Ziv estimator.
        
        :param location_series: List of locations in the time series.
        :return: Estimated entropy value.
        """
        n = len(location_series)
        if n == 0:
            return 0

        # Initializing the entropy estimate
        S_est = 0

        # Iterating over the location series to compute entropy
        for i in range(n):
            substring_length = 1

            while substring_length <= i + 1:
                substring = location_series[i:i + substring_length]
                # Check if the substring appears before the current position
                check_string = [location_series[j:j + substring_length] for j in range(i - substring_length + 1)]
                if substring in check_string:
                    substring_length += 1
                else:
                    break

            S_est += substring_length

        return  n / S_est * math.log2(n)
    
    # Group the DataFrame by 'user' and aggregate 'pos' into lists
    user_trajectories = df.groupby('user')['pos'].agg(lambda x: list(x))

    # Calculate entropy for each user
    return user_trajectories.apply(lempel_ziv_entropy)


from scipy.optimize import brentq

def calculate_predictability(N, S_i):
    """
    Find the root of the equation 
    0 = -[x * log2(x) + (1 - x) * log2(1 - x)] + (1 - x) * log2(N - 1) - S_i
    """
    def equation(x):
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x) + (1 - x) * np.log2(N - 1) - S_i

    # Find the root using Brent's method
    # We use a range between 1e-5 and 1 - 1e-5 to avoid the endpoints where the log function is undefined
    
    try:
        root = brentq(equation,1e-5, 1 - 1e-5)
        return root
    except ValueError:
        # If the equation has no root, return 0
        return 0

