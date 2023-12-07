import pandas as pd
from datetime import datetime, timedelta
import random

def generate_synthetic_data(num_users=100, num_days=10):
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


def generate_synthetic_data_random_loc(num_users=100, num_days=10):
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
        for day in range(num_days):
            date = start_date + timedelta(days=day)
            user_week = f'u{user_id}_w{date.isocalendar()[1]}'

            # Alternate between the two locations
            pos = user_locations[user_id][day % 2]
            rank = location_ranks[user_id][pos]

            synthetic_data.append([user_id, date, user_week, pos, rank])

    columns = ['user', 'date', 'user_week', 'pos', 'rank']
    return pd.DataFrame(synthetic_data, columns=columns)


