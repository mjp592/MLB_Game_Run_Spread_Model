import pandas as pd
import multiprocessing
import numpy as np


def calculate_pitcher_stats(df, begin_index, last_index, min_innings, date, pitcher):
    temp_innings = df.loc[begin_index:last_index, 'innings_pitched'].sum()

    if temp_innings >= min_innings:
        temp_runs = df.loc[begin_index:last_index, 'runs_scored'].sum()
        temp_outs = df.loc[begin_index:last_index, 'out_flag'].sum()
        temp_strikeout = df.loc[begin_index:last_index, 'strikeout_flag'].sum()
        temp_walks = df.loc[begin_index:last_index, 'walk_flag'].sum()
        temp_hits = df.loc[begin_index:last_index, 'hits'].sum()
        temp_batters = df.loc[begin_index:last_index, 'batters_faced'].sum()
        temp_total_bases = df.loc[begin_index:last_index, 'total_bases'].sum()
        temp_single = df.loc[begin_index:last_index, 'single_flag'].sum()
        temp_double = df.loc[begin_index:last_index, 'double_flag'].sum()
        temp_triple = df.loc[begin_index:last_index, 'triple_flag'].sum()
        temp_home_run = df.loc[begin_index:last_index, 'home_run_flag'].sum()
        temp_hbp = df.loc[begin_index:last_index, 'hit_by_pitch_flag'].sum()
        temp_double_play = df.loc[begin_index:last_index, 'double_play_flag'].sum()

        temp_era = (temp_runs / temp_innings) * 9
        temp_whip = (temp_hits + temp_walks) / temp_innings
        temp_tbip = temp_total_bases / temp_innings
        temp_out_rate = temp_outs / temp_batters
        temp_k_rate = temp_strikeout / temp_batters
        temp_w_rate = temp_walks / temp_batters
        temp_single_rate = temp_single / temp_batters
        temp_double_rate = temp_double / temp_batters
        temp_triple_rate = temp_triple / temp_batters
        temp_hr_rate = temp_home_run / temp_batters
        temp_hbp_rate = temp_hbp / temp_batters
        temp_dp_rate = temp_double_play / temp_batters

        cum_prob = temp_out_rate + temp_k_rate + temp_w_rate + temp_single_rate + temp_double_rate + temp_triple_rate + temp_hr_rate + temp_hbp_rate + temp_dp_rate

        temp_out_rate = temp_out_rate / cum_prob
        temp_k_rate = temp_k_rate / cum_prob
        temp_w_rate = temp_w_rate / cum_prob
        temp_single_rate = temp_single_rate / cum_prob
        temp_double_rate = temp_double_rate / cum_prob
        temp_triple_rate = temp_triple_rate / cum_prob
        temp_hr_rate = temp_hr_rate / cum_prob
        temp_hbp_rate = temp_hbp_rate / cum_prob
        temp_dp_rate = temp_dp_rate / cum_prob

        temp_row = [date, pitcher, temp_era, temp_whip, temp_tbip, temp_k_rate, temp_w_rate, temp_single_rate
                    , temp_double_rate, temp_triple_rate, temp_hr_rate, temp_hbp_rate, temp_dp_rate
                    , temp_out_rate, temp_innings]

        return temp_row

    return []


def generate_single_pitcher_prior_pitching_stats(df, date_list, pitcher_id, min_innings, maximum_innings):
    temp_df = df[(df['pitcher'] == pitcher_id)].copy()
    temp_df = temp_df[(temp_df['game_type'] == 'R')].reset_index(drop=True)

    df = df[(df['game_type'] == 'R')].reset_index(drop=True)

    temp_list = []
    result = []
    for d in date_list:

        temp_df = temp_df[(temp_df['game_date'] < d)]

        num_appearances = temp_df.shape[0]
        last_index = num_appearances - 1

        for i in range(1, num_appearances):

            begin_index = num_appearances - i

            result = calculate_pitcher_stats(temp_df, begin_index, last_index, min_innings, d, pitcher_id)

            if len(result) != 0:
                if i != last_index:

                    if result[12] >= maximum_innings:
                        result.append(0)
                        temp_list.append(result)
                        break

                if i == last_index:
                    result.append(0)
                    temp_list.append(result)
                    break

        if len(result) != 0:
            continue

        rows = df.shape[0]
        end_index = rows - 1

        result = calculate_pitcher_stats(df, 0, end_index, min_innings, d, pitcher_id)

        result.append(1)
        temp_list.append(result)

    return temp_list


def pitcher_stat_mapping(temp_pitcher_id):

    temp_df = bat_df[(bat_df['pitcher'] == temp_pitcher_id)].copy()

    unique_dates = np.unique(temp_df['game_date'])
    unique_dates = np.flip(unique_dates)

    result = generate_single_pitcher_prior_pitching_stats(pit_df, unique_dates, temp_pitcher_id
                                                          , minimum_innings, max_innings)

    return result


def init_pitcher_stat_mapping(batter_df, pitch_df, min_innings, maximum_innings):
    global bat_df
    global pit_df
    global minimum_innings
    global max_innings

    bat_df = batter_df
    pit_df = pitch_df
    minimum_innings = min_innings
    max_innings = maximum_innings

    return


def parallel_pitcher_stat_mapping(batter_df, pitch_df, min_innings, maximum_innings, num_of_workers, filename):
    agg_df_columns = ['game_date', 'pitcher', 'era', 'whip', 'tbip', 'strikeout_rate', 'walk_rate', 'single_rate'
                      , 'double_rate', 'triple_rate', 'home_run_rate', 'hit_by_pitch_rate', 'double_play_rate'
                      , 'out_rate', 'num_innings', 'avg_value?']

    temp_list = []

    unique_pitchers = np.unique(batter_df['pitcher'])
    i = 0
    with multiprocessing.Pool(processes=num_of_workers, initializer=init_pitcher_stat_mapping
                                                      , initargs=(batter_df, pitch_df
                                                                  , min_innings, maximum_innings)) as pool:

        for result in pool.imap(pitcher_stat_mapping, unique_pitchers):
            print(i)
            i = i + 1
            temp_list = temp_list + result

    temp_df = pd.DataFrame(temp_list, columns=agg_df_columns)
    final_df = batter_df.merge(temp_df, how='left', on=['game_date', 'pitcher'])

    final_df = final_df.dropna().reset_index(drop=True)

    # final_df.to_csv('check.csv', index=False)
    final_df.to_csv(filename, index=False)

    return final_df


if __name__ == '__main__':
    num_workers = multiprocessing.cpu_count()

    # run code to execute batter to pitcher mapping ------------------------------------------------------------------

    batting_df = pd.read_csv('batter_plate_appearance_outcomes.csv')
    print(batting_df)
    batting_df = batting_df[(batting_df['game_type'] == 'R')].reset_index(drop=True)
    pitching_df = pd.read_csv('pitcher_appearance_outcomes.csv')
    mapped_df = parallel_pitcher_stat_mapping(batting_df, pitching_df, 50, 200
                                              , num_workers, 'batter_outcome_and_prior_pitching_stats.csv')
