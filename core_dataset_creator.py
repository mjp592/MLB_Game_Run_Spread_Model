import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pybaseball.cache
from pybaseball import statcast
from pybaseball import playerid_lookup


def one_hot_string_encoding(df, existing_label, matching_string, new_label):
    df.loc[df[existing_label] == matching_string, new_label] = 1
    df.loc[df[existing_label] != matching_string, new_label] = 0

    return df


def transform_and_remove_unwanted_events(df, label, remove_iterable, remap_iterable, new_value_iterable):

    if len(remap_iterable) != len(new_value_iterable):
        print('Each remapping value does not have a direct mapping to a replacement value.')
        return df

    for k in remove_iterable:

        df = df[(df[label] != k)]

    df = df.reset_index(drop=True)

    for i in range(0, len(remap_iterable)):
        old_value = remap_iterable[i]
        new_value = new_value_iterable[i]
        df.loc[df[label] == old_value, label] = new_value

    return df


def create_all_batter_event_codings(df, existing_label, event_list):

    for t in event_list:
        new_label = t + "_flag"
        df = one_hot_string_encoding(df, existing_label, t, new_label)

    return df


def download_raw_pitch_data(filename):
    pybaseball.cache.enable()

    today = datetime.today().strftime('%Y-%m-%d')

    df = statcast(start_dt='2008-01-01', end_dt=today)
    print(df)
    df.to_csv(filename, index=False)

    return df


def generate_batter_plate_appearance_data(filename, df, pa_label, remove_iterable
                                          , remap_iterable, new_iterable, pa_headers):

    df = df.filter(pa_headers).dropna().reset_index(drop=True)

    df = transform_and_remove_unwanted_events(df, pa_label, remove_iterable, remap_iterable, new_iterable)

    unique_events = np.unique(df[pa_label])
    print(unique_events)

    df = create_all_batter_event_codings(df, pa_label, unique_events)

    df.to_csv(filename, index=False)

    print(df)

    return df


if __name__ == '__main__':

    # playerid_lookup('kershaw', 'clayton')

    batter_headers = ['game_date', 'game_type', 'batter', 'pitcher', 'events'
                      , 'p_throws', 'home_team', 'game_pk']

    removed_events_list = ['catcher_interf', 'caught_stealing_2b'
        , 'caught_stealing_3b', 'caught_stealing_home', 'field_error'
        , 'pickoff_3b', 'pickoff_caught_stealing_2b', 'sac_bunt'
        , 'sac_bunt_double_play', 'triple_play', 'game_advisory', 'passed_ball'
        , 'pickoff_1b', 'pickoff_2b', 'pickoff_caught_stealing_3b'
        , 'pickoff_caught_stealing_home', 'stolen_base_2b', 'wild_pitch'
        , 'ejection', 'other_advance', 'pickoff_error_2b', 'stolen_base_3b'
        , 'stolen_base_home']

    remap_list = ['grounded_into_double_play'
        , 'field_out', 'fielders_choice', 'fielders_choice_out', 'force_out'
        , 'other_out', 'sac_fly', 'sac_fly_double_play'
        , 'strikeout_double_play', 'intent_walk', 'runner_double_play']

    new_mapping_list = ['double_play', 'out', 'out', 'out', 'out'
        , 'out', 'out', 'double_play', 'double_play', 'walk', 'double_play']

    data = pd.read_csv('raw_dataset.csv', nrows=20000)
    # data = pd.read_csv('raw_dataset.csv', low_memory=False)



    # data = generate_batter_plate_appearance_data('batter_plate_apperance_outcomes.csv', data, 'events'
    #                                             , removed_events_list, remap_list, new_mapping_list, batter_headers)


