"""
Synthetic Data Generation 

This script contains functions to generate synthetic data for masked location prediction.
The generated data mimics various patterns of user behavior, including alternating locations, 
a constant sequence of locations, weekly location cycles, random distinct locations, and constant distinct locations.

Functions:
- alternating_loc_constant: Generates synthetic data with users alternating between two constant locations.
- ten_sequence_loc_constant: Generates synthetic data with users cycling through a sequence of ten constant locations.
- weekly_location_cycle: Generates synthetic data with users cycling through seven locations based on the day of the week.
- random_loc_distinct: Generates synthetic data with users alternating between two random distinct locations.
- constant_loc_distinct: Generates synthetic data with users consistently visiting unique locations.
"""

import pandas as pd
from datetime import datetime, timedelta
import random

def alternating_loc_constant(num_users=100, num_days=10):
    synthetic_data = []

    start_date = datetime.now().date()
    user_locations = {user_id: [(1, 1), (2, 2)] for user_id in range(num_users)}
    location_ranks = {user_id: {user_locations[user_id][0]: 1, user_locations[user_id][1]: 2} for user_id in range(num_users)}

    for user_id in range(num_users):
        for day in range(num_days):
            date = start_date + timedelta(days=day)
            user_week = f'u{user_id}_w{date.isocalendar()[1]}'

            # Alternate between the two locations
            pos = user_locations[user_id][day % 2]
            rank = location_ranks[user_id][pos]

            synthetic_data.append([user_id, date, user_week, pos, rank])

    columns = ['user', 'date', 'user_week', 'pos', 'rank']
    return pd.DataFrame(synthetic_data, columns=columns)


def ten_sequence_loc_constant(num_users=100, num_days=10):
    synthetic_data = []

    start_date = datetime.now().date()

    # Define a common list of 10 locations
    common_locations = [(i, i + 1) for i in range(1, 11)]

    # Assign ranks to these locations
    location_ranks = {common_locations[i]: i + 1 for i in range(len(common_locations))}

    for user_id in range(num_users):
        current_week = -1
        location_index = 0

        for day in range(num_days):
            date = start_date + timedelta(days=day)
            week_number = date.isocalendar()[1]

            # Check if the week has changed
            if week_number != current_week:
                current_week = week_number
                location_index = 0  # Reset location index for new week

            user_week = f'u{user_id}_w{week_number}'
            pos = common_locations[location_index % len(common_locations)]
            rank = location_ranks[pos]

            synthetic_data.append([user_id, date, user_week, pos, rank])

            location_index += 1  # Move to the next location for the next day

    columns = ['user', 'date', 'user_week', 'pos', 'rank']
    return pd.DataFrame(synthetic_data, columns=columns)



def weekly_location_cycle(num_users=100, num_days=70, grid_size = 200):  # Example: 10 weeks
    synthetic_data = []

    start_date = datetime.now().date()

    # Generate 7 unique locations for each user
    user_locations = {user_id: [(random.randint(1, grid_size), random.randint(1, grid_size)) for _ in range(7)] 
                      for user_id in range(num_users)}
    
    # Assign ranks to these locations in increasing order
    location_ranks = {user_id: {loc: idx + 1 for idx, loc in enumerate(locs)} for user_id, locs in user_locations.items()}


    for user_id in range(num_users):
        for day in range(num_days):
            date = start_date + timedelta(days=day)
            user_week = f'u{user_id}_w{date.isocalendar()[1]}'

            # Get location based on the day of the week
            day_of_week = date.weekday()  # Monday is 0 and Sunday is 6
            pos = user_locations[user_id][day_of_week]
            rank = location_ranks[user_id][pos]

            synthetic_data.append([user_id, date, user_week, pos, rank])

    columns = ['user', 'date', 'user_week', 'pos', 'rank']
    return pd.DataFrame(synthetic_data, columns=columns)




def random_loc_distinct(num_users=100, num_days=10):
    synthetic_data = []

    start_date = datetime.now().date()

    # Generate two random locations for each user
    user_locations = {user_id: [(random.randint(1, 200), random.randint(1, 200)), 
                                (random.randint(1, 200), random.randint(1, 200))] 
                      for user_id in range(num_users)}

    # Assign ranks to locations for consistency
    location_ranks = {user_id: {user_locations[user_id][0]: random.randint(1, 5), 
                                user_locations[user_id][1]: random.randint(6, 10)} 
                      for user_id in range(num_users)}

    for user_id in range(num_users):
        current_week = -1
        location_index = 0

        for day in range(num_days):
            date = start_date + timedelta(days=day)
            week_number = date.isocalendar()[1]

            # Check if the week has changed
            if week_number != current_week:
                current_week = week_number
                location_index = 0  # Reset location index for new week

            user_week = f'u{user_id}_w{week_number}'

            # Alternate between the two locations, starting with the first at each new week
            pos = user_locations[user_id][location_index % 2]
            rank = location_ranks[user_id][pos]

            synthetic_data.append([user_id, date, user_week, pos, rank])

            location_index += 1  # Move to the next location for the next day

    columns = ['user', 'date', 'user_week', 'pos', 'rank']
    return pd.DataFrame(synthetic_data, columns=columns)




def constant_loc_distinct(num_users=100, num_days=10):
    synthetic_data = []

    start_date = datetime.now().date()
    
    # Generate a unique location for each user
    user_locations = {user_id: (random.randint(1, 10), random.randint(1, 10)) 
                      for user_id in range(num_users)}
    
    # Assign rank 1 for each location
    location_ranks = {loc: 1 for loc in user_locations.values()}

    for user_id in range(num_users):
        for day in range(num_days):
            date = start_date + timedelta(days=day)
            user_week = f'u{user_id}_w{date.isocalendar()[1]}'

            pos = user_locations[user_id]
            rank = location_ranks[pos]

            synthetic_data.append([user_id, date, user_week, pos, rank])

    columns = ['user', 'date', 'user_week', 'pos', 'rank']
    return pd.DataFrame(synthetic_data, columns=columns)


