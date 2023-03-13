import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from general_functions import transform_and_remove_unwanted_events
from general_functions import one_hot_string_encoding


def create_fielder_event_table(df, number_label, prefix, event_label, kept_labels):

    rows = df.shape[0]

    temp_list = []
    for i in range(0, rows):
        temp_row = []

        temp_number = str(int(df.loc[i, number_label]))
        temp_label = prefix + temp_number

        if temp_label == (prefix + '1'):
            temp_label = 'pitcher'

        temp_date = df.loc[i, 'game_date']
        temp_id = df.loc[i, temp_label]
        temp_error_flag = df.loc[i, event_label]

        temp_row.append(temp_date)
        temp_row.append(temp_id)
        temp_row.append(temp_error_flag)

        temp_list.append(temp_row)

    result_df = pd.DataFrame(temp_list, columns=kept_labels)
    result_df = result_df

    return result_df


if __name__ == '__main__':

    raw_df = pd.read_csv('raw_dataset.csv', low_memory=False)

    raw_df = raw_df.filter(['game_date', 'events', 'hit_location', 'pitcher', 'fielder_2', 'fielder_3', 'fielder_4'
                            , 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9'])

    raw_df = raw_df.dropna().reset_index(drop=True)

    ['catcher_interf', 'caught_stealing_2b', 'caught_stealing_3b'
     , 'caught_stealing_home', 'double', 'double_play', 'ejection', 'field_error'
     , 'field_out', 'fielders_choice', 'fielders_choice_out', 'force_out'
     , 'game_advisory', 'grounded_into_double_play', 'home_run', 'intent_walk'
     , 'other_advance', 'other_out', 'passed_ball', 'pickoff_1b', 'pickoff_2b'
     , 'pickoff_3b', 'pickoff_caught_stealing_2b', 'pickoff_caught_stealing_3b'
     , 'pickoff_caught_stealing_home', 'pickoff_error_2b', 'runner_double_play'
     , 'sac_bunt', 'sac_bunt_double_play', 'sac_fly', 'sac_fly_double_play'
     , 'single', 'stolen_base_2b', 'stolen_base_3b', 'stolen_base_home', 'strikeout'
     , 'strikeout_double_play', 'triple', 'triple_play', 'walk', 'wild_pitch']

    remove_list = ['catcher_interf', 'game_advisory', 'passed_ball', 'wild_pitch', 'ejection', 'other_advance']

    remap_list = ['field_error', 'pickoff_error_2b']
    new_list = ['error', 'error']

    raw_df = transform_and_remove_unwanted_events(raw_df, 'events', remove_list, remap_list, new_list)

    raw_df = one_hot_string_encoding(raw_df, 'events', 'error', 'error_flag')

    fielder_labels = ['game_date', 'fielder', 'error_flag']

    fielder_df = create_fielder_event_table(raw_df, 'hit_location', 'fielder_', 'error_flag', fielder_labels)
    fielder_df = fielder_df.sort_values(by='game_date').reset_index(drop=True)

    fielder_df.to_csv('fielder_play_chance_outcome.csv', index=False)

    print(fielder_df)
    print(np.sum(fielder_df['error_flag']))
