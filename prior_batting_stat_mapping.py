import pandas as pd
import multiprocessing
import numpy as np


def batter_stat_prior_calc(temp_batter):

    temp_df = batter_prior_df[(batter_prior_df['batter'] == temp_batter)].copy().reset_index(drop=True)

    unique_dates = np.unique(temp_df['game_date'])
    unique_dates = np.flip(unique_dates)
    temp_list = []

    for d in unique_dates:
        temp_df = temp_df[(temp_df['game_date'] < d)].copy().reset_index(drop=True)
        temp_rows = temp_df.shape[0]
        temp_result = [d, temp_batter]
        average_flag = 0

        for x in agg_batter_prior_columns:
            if temp_rows > max_plate_appearances:
                temp_value = temp_df.loc[0:max_plate_appearances, x].sum() / max_plate_appearances
            elif temp_rows < min_plate_appearances:
                temp_value = batter_prior_df[x].sum() / batter_prior_df.shape[0]
                average_flag = 1
            else:
                temp_value = temp_df.loc[0:temp_rows, x].sum() / temp_rows

            temp_result.append(temp_value)
        temp_result.append(average_flag)
        temp_list.append(temp_result)

    return temp_list


def init_batter_stat_prior(df, agg_columns, min_pa, max_pa):
    global batter_prior_df
    global agg_batter_prior_columns
    global max_plate_appearances
    global min_plate_appearances

    batter_prior_df = df
    agg_batter_prior_columns = agg_columns
    max_plate_appearances = max_pa
    min_plate_appearances = min_pa

    return


def parallel_batter_stat_mapping(df, agg_columns, batter_stat_columns, min_pa, max_pa, num_of_workers):

    temp_list = []
    i = 0
    with multiprocessing.Pool(processes=num_of_workers, initializer=init_batter_stat_prior
                                                      , initargs=(df, agg_columns, min_pa, max_pa)) as pool:

        for result in pool.imap(batter_stat_prior_calc, np.unique(df['batter'])):
            print(i)
            i = i + 1
            temp_list = temp_list + result

    result_df = pd.DataFrame(temp_list, columns=batter_stat_columns)

    final_df = df.merge(result_df, how='left', on=['game_date', 'batter'])

    final_df = final_df.dropna().reset_index(drop=True)

    return final_df


if __name__ == '__main__':
    num_workers = multiprocessing.cpu_count()

    data = pd.read_csv('batter_outcome_and_prior_pitching_stats.csv')
    data = data[(data['avg_value?'] == 0)].reset_index(drop=True)

    batter_stat_agg_columns = ['out_flag', 'strikeout_flag', 'walk_flag', 'single_flag', 'double_flag', 'triple_flag'
                               , 'home_run_flag', 'double_play_flag', 'hit_by_pitch_flag']

    batter_stat_headers = ['game_date', 'batter', 'batter_out_rate', 'batter_strikeout_rate', 'batter_walk_rate'
                           , 'batter_single_rate', 'batter_double_rate', 'batter_triple_rate', 'batter_home_run_rate'
                           , 'batter_double_play_rate', 'batter_hit_by_pitch_flag', 'average_batter?']

    export_df = parallel_batter_stat_mapping(data, batter_stat_agg_columns, batter_stat_headers, 200, 600, num_workers)
    export_df.to_csv('batter_outcome_and_prior_pitching_batting_stats.csv', index=False)
