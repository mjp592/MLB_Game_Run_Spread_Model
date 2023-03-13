import random
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np
import scipy as scp
from pybaseball import team_game_logs
from pybaseball import statcast_single_game
from statsapi import schedule
from statsapi import lookup_player
from statsapi import roster
from pybaseball.lahman import appearances
import requests
import json


def calculate_sim_pitcher_stats(df, begin_index, last_index, min_innings, date, pitcher):
    temp_innings = df.loc[begin_index:last_index, 'innings_pitched'].sum()

    if temp_innings >= min_innings:
        temp_avg_innings = df.loc[begin_index:last_index, 'innings_pitched'].mean()
        temp_std_innings = df.loc[begin_index:last_index, 'innings_pitched'].std()
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

        temp_row = [date, pitcher, temp_avg_innings, temp_std_innings, temp_era, temp_whip, temp_tbip, temp_k_rate
                    , temp_w_rate, temp_single_rate, temp_double_rate, temp_triple_rate, temp_hr_rate, temp_hbp_rate
                    , temp_dp_rate, temp_out_rate, temp_innings]

        return temp_row

    return []


def sim_single_pitcher_prior_pitching_stats(df, current_date, pitcher_id, min_innings, maximum_innings):
    temp_df = df[(df['pitcher'] == pitcher_id)].copy()
    temp_df = temp_df[(temp_df['game_type'] == 'R')].reset_index(drop=True)

    df = df[(df['game_type'] == 'R')].reset_index(drop=True)
    df = df[(df['innings_pitched'] > 2)].reset_index(drop=True)

    result = []

    temp_df = temp_df[(temp_df['game_date'] < current_date)]

    num_appearances = temp_df.shape[0]
    last_index = num_appearances - 1

    for i in range(1, num_appearances):

        begin_index = num_appearances - i

        result = calculate_sim_pitcher_stats(temp_df, begin_index, last_index, min_innings, current_date, pitcher_id)

        if len(result) != 0:
            if i != last_index:

                if result[16] >= maximum_innings:
                    result.append(0)
                    break

            if i == last_index:
                result.append(0)
                break

    if len(result) == 0:
        rows = df.shape[0]
        end_index = rows - 1

        result = calculate_sim_pitcher_stats(df, 0, end_index, min_innings, current_date, pitcher_id)

        result.append(1)

    return result


def get_lineups_for_game(game_id, year):

    string = str(game_id)

    url = "https://statsapi.mlb.com/api/v1/schedule?gamePk=" + string + "&language=en&hydrate=lineups"

    result = requests.get(url)

    raw_content = result.content
    raw_dict = json.loads(raw_content.decode('utf-8'))
    first_level_dict = raw_dict['dates'][0]
    second_level_dict = first_level_dict['games'][0]
    third_level_dict = second_level_dict['lineups']

    home_list = third_level_dict['homePlayers']
    away_list = third_level_dict['awayPlayers']

    home_lineup = pd.DataFrame(home_list).filter(['id', 'fullName'])
    away_lineup = pd.DataFrame(away_list).filter(['id', 'fullName'])

    game = schedule(game_id=string)[0]

    home_pitcher_name = game['home_probable_pitcher']
    away_pitcher_name = game['away_probable_pitcher']
    game_date = game['game_date']

    home_team_id = game['home_id']
    away_team_id = game['away_id']

    home_pitcher_dict = lookup_player(home_pitcher_name, season=year)[0]
    away_pitcher_dict = lookup_player(away_pitcher_name, season=year)[0]
    print(home_pitcher_dict)
    home_pitcher_df = pd.DataFrame(home_pitcher_dict).filter(['id', 'fullName']).reset_index(drop=True)
    away_pitcher_df = pd.DataFrame(away_pitcher_dict).filter(['id', 'fullName']).reset_index(drop=True)

    # print(home_lineup)
    # print(away_lineup)
    # print(home_pitcher_df)
    # print(away_pitcher_df)

    final_result = (home_lineup, away_lineup, home_pitcher_df, away_pitcher_df, game_date, home_team_id, away_team_id)

    return final_result


def calculate_bullpen_stats(team_id, current_date, pitch_df, min_innings, max_innings, column_headers):
    temp_roster = roster(team_id, date=current_date)

    roster_df = pd.DataFrame([x[3:].strip() for x in temp_roster.split('\n')])
    second_df = pd.DataFrame([[x[0:2].strip(), x[2:].strip()] for x in roster_df[0]], columns=['Position', 'Name'])
    second_df = second_df[(second_df['Position'] == 'P')].reset_index(drop=True)

    year = current_date[0:4]

    temp_list = []

    for n in second_df['Name']:
        pitcher_dict = lookup_player(n, season=year)[0]
        pitcher_df = pd.DataFrame(pitcher_dict).filter(['id', 'fullName']).reset_index(drop=True)
        temp_id = pitcher_df.loc[0, 'id']
        stats = sim_single_pitcher_prior_pitching_stats(pitch_df, current_date, temp_id, min_innings, max_innings)
        temp_list.append(stats)

    temp_df = pd.DataFrame(temp_list, columns=column_headers)

    second_df = pd.concat([second_df, temp_df], axis=1)

    second_df = second_df[(second_df['Average Innings'] <=2)].reset_index(drop=True)
    summary = second_df.describe()

    bullpen_stats = summary.loc[['mean', 'std'], :]

    print(second_df)
    print(bullpen_stats)
    print(summary)
    return bullpen_stats


def stat_prior_calc(df, temp_batter, current_date, agg_columns, min_pa, max_pa, player_id_label, date_label):

    temp_df = df[(df[player_id_label] == temp_batter)].copy().reset_index(drop=True)

    temp_df = temp_df[(temp_df[date_label] < current_date)].copy().reset_index(drop=True)
    temp_rows = temp_df.shape[0]
    temp_result = [temp_batter]

    for x in agg_columns:
        if temp_rows > max_pa:
            temp_value = temp_df.loc[0:max_pa, x].sum() / max_pa
        elif temp_rows < min_pa:
            temp_value = df[x].sum() / df.shape[0]
        else:
            temp_value = temp_df.loc[0:temp_rows, x].sum() / temp_rows

        temp_result.append(temp_value)

    return temp_result


def calculate_lineup_stats(df, lineup, index_date, agg_columns, columns_names
                           , min_pa, max_pa, player_id_label, date_label):

    temp_list = []

    for i in lineup['id']:
        result = stat_prior_calc(df, i, index_date, agg_columns, min_pa, max_pa, player_id_label, date_label)
        temp_list.append(result)

    result_df = pd.DataFrame(temp_list, columns=columns_names)
    final_df = lineup.merge(result_df, how='left', on='id')

    print(final_df)

    return final_df


def simulate_game(home_batting_stats, away_batting_stats, home_fielding_stats, away_fielding_stats
                  , home_sp_stats, away_sp_stats, home_bp_stats, away_bp_stats, model_iterable, scalar_iterable):

    home_sp_avg_innings = home_sp_stats.loc[0, 'Average Innings']
    home_sp_std_innings = home_sp_stats.loc[0, 'Std Innings']

    away_sp_avg_innings = away_sp_stats.loc[0, 'Average Innings']
    away_sp_std_innings = away_sp_stats.loc[0, 'Std Innings']

    home_bp_avg_innings = home_bp_stats.loc['mean', 'Average Innings']
    home_bp_std_innings = home_bp_stats.loc['mean', 'Std Innings']

    away_bp_avg_innings = away_bp_stats.loc['mean', 'Average Innings']
    away_bp_std_innings = away_bp_stats.loc['mean', 'Std Innings']

    home_sp_stats = home_sp_stats.drop(['Average Innings', 'Std Innings'], axis=1)
    away_sp_stats = away_sp_stats.drop(['Average Innings', 'Std Innings'], axis=1)

    home_bp_stats = home_bp_stats.drop(['Average Innings', 'Std Innings'], axis=1)
    away_bp_stats = away_bp_stats.drop(['Average Innings', 'Std Innings'], axis=1)

    home_error_rate = home_fielding_stats['error_rate'].mean()
    away_error_rate = away_fielding_stats['error_rate'].mean()

    home_batter_index = 0
    away_batter_index = 0

    home_score = 0
    away_score = 0

    inning = 1
    outs = 0
    first_base = False
    second_base = False
    third_base = False

    home_lineup_regression_inputs = pd.concat([away_sp_stats, home_batting_stats], axis=1).ffill()
    away_lineup_regression_inputs = pd.concat([home_sp_stats, away_batting_stats], axis=1).ffill()

    current_inputs = away_lineup_regression_inputs.loc[0, :].values.reshape(1, -1)

    temp_outcome = simulate_plate_appearance(current_inputs, model_iterable, scalar_iterable)

    return


def generate_plate_appearance_distribution(inputs, models, scalars):
    raw_results = []
    length = len(models)

    for i in range(0, length):
        temp_model = models[i]
        temp_scalar = scalars[i]
        temp_data = temp_scalar.transform(inputs)
        temp_result = temp_model.predict_proba(temp_data)[0][1]
        raw_results.append(temp_result)

    normalized_results = raw_results / sum(raw_results)

    return normalized_results


def simulate_plate_appearance(inputs, models, scalars):
    distribution = generate_plate_appearance_distribution(inputs, models, scalars)

    outcome_name_list = ['out', 'walk', 'strikeout', 'single', 'double', 'triple', 'home_run', 'double_play', 'hbp']
    lb = 0
    hb = 0

    number_possible_outcomes = len(distribution)

    random_sample = random.random()

    for i in range(number_possible_outcomes):
        temp_prob = distribution[i]
        if i == 0:
            hb = hb + temp_prob
        elif i == (number_possible_outcomes - 1):
            lb = hb
            hb = 1
        else:
            lb = hb
            hb = hb + temp_prob

        if lb <= random_sample:
            if hb > random_sample:
                result = outcome_name_list[i]
                break

    return result


if __name__ == '__main__':

    minimum_allowable_innings = 50
    maximum_allowable_innings = 200

    minimum_allowable_plate_appearances = 200
    maximum_allowable_plate_appearances = 600

    minimum_allowable_fielding_plays = 50
    maximum_allowable_fielding_plays = 250

    pitching_df = pd.read_csv('pitcher_appearance_outcomes.csv')
    plate_appearance_df = pd.read_csv('batter_plate_appearance_outcomes.csv')
    fielding_df = pd.read_csv('fielder_play_chance_outcome.csv')
    fielding_df = fielding_df.sort_values(['game_date'], ascending=False).reset_index(drop=True)

    out_model = joblib.load('models/out_logistic_regression_model.joblib')
    walk_model = joblib.load('models/walk_logistic_regression_model.joblib')
    strikeout_model = joblib.load('models/strikeout_logistic_regression_model.joblib')
    single_model = joblib.load('models/single_logistic_regression_model.joblib')
    double_model = joblib.load('models/double_logistic_regression_model.joblib')
    triple_model = joblib.load('models/triple_logistic_regression_model.joblib')
    home_run_model = joblib.load('models/home_run_logistic_regression_model.joblib')
    double_play_model = joblib.load('models/double_play_logistic_regression_model.joblib')
    hit_by_pitch_model = joblib.load('models/hit_by_pitch_logistic_regression_model.joblib')

    model_list = [out_model, walk_model, strikeout_model, single_model, double_model
                  , triple_model, home_run_model, double_play_model, hit_by_pitch_model]

    out_scaler = joblib.load('scalers/out_min_max_scaler.joblib')
    walk_scaler = joblib.load('scalers/walk_min_max_scaler.joblib')
    strikeout_scaler = joblib.load('scalers/strikeout_min_max_scaler.joblib')
    single_scaler = joblib.load('scalers/single_min_max_scaler.joblib')
    double_scaler = joblib.load('scalers/double_min_max_scaler.joblib')
    triple_scaler = joblib.load('scalers/triple_min_max_scaler.joblib')
    home_run_scaler = joblib.load('scalers/home_run_min_max_scaler.joblib')
    double_play_scaler = joblib.load('scalers/double_play_min_max_scaler.joblib')
    hit_by_pitch_scaler = joblib.load('scalers/hit_by_pitch_min_max_scaler.joblib')

    scaler_list = [out_scaler, walk_scaler, strikeout_scaler, single_scaler, double_scaler
                   , triple_scaler, home_run_scaler, double_play_scaler, hit_by_pitch_scaler]

    batter_stat_agg_columns = ['out_flag', 'double_play_flag', 'hit_by_pitch_flag', 'strikeout_flag', 'walk_flag'
                               , 'single_flag', 'double_flag', 'triple_flag', 'home_run_flag']

    batter_stat_headers = ['id', 'batter_out_rate', 'batter_double_play_rate', 'batter_hit_by_pitch_rate'
                           , 'batter_strikeout_rate', 'batter_walk_rate', 'batter_single_rate', 'batter_double_rate'
                           , 'batter_triple_rate', 'batter_home_run_rate']

    pitching_stat_headers = ['game_date', 'pitcher', 'Average Innings', 'Std Innings', 'era', 'whip', 'tbip'
                             , 'strikeout_rate', 'walk_rate', 'single_rate', 'double_rate', 'triple_rate'
                             , 'home_run_rate', 'hit_by_pitch_rate', 'double_play_rate', 'out_rate', 'num_innings'
                             , 'avg_value?']

    fielder_stat_agg_columns = ['error_flag']
    fielder_stat_headers = ['id', 'error_rate']

    games = schedule(start_date='07/01/2018', end_date='07/01/2018')
    # print(appearances())
    for g in games:
        temp_game_id = g['game_id']

        home_lu, away_lu, home_pitcher, away_pitcher, date, home_id, away_id = get_lineups_for_game(temp_game_id, 2018)

        home_bullpen_stats = calculate_bullpen_stats(home_id, date, pitching_df, minimum_allowable_innings
                                                     , maximum_allowable_innings, pitching_stat_headers)
        home_bullpen_stats = home_bullpen_stats.drop(['pitcher', 'num_innings', 'avg_value?'], axis=1)

        away_bullpen_stats = calculate_bullpen_stats(away_id, date, pitching_df, minimum_allowable_innings
                                                     , maximum_allowable_innings, pitching_stat_headers)
        away_bullpen_stats = away_bullpen_stats.drop(['pitcher', 'num_innings', 'avg_value?'], axis=1)

        full_home_lineup_batting = calculate_lineup_stats(plate_appearance_df, home_lu, date, batter_stat_agg_columns
                                                          , batter_stat_headers, minimum_allowable_plate_appearances
                                                          , maximum_allowable_plate_appearances, 'batter', 'game_date')
        full_home_lineup_batting = full_home_lineup_batting.drop(['id', 'fullName'], axis=1)

        full_away_lineup_batting = calculate_lineup_stats(plate_appearance_df, away_lu, date, batter_stat_agg_columns
                                                          , batter_stat_headers, minimum_allowable_plate_appearances
                                                          , maximum_allowable_plate_appearances, 'batter', 'game_date')
        full_away_lineup_batting = full_away_lineup_batting.drop(['id', 'fullName'], axis=1)

        home_pitcher_id = home_pitcher.loc[0, 'id']
        away_pitcher_id = away_pitcher.loc[0, 'id']

        home_starting_pitcher_stats = sim_single_pitcher_prior_pitching_stats(pitching_df, date, home_pitcher_id
                                                                              , minimum_allowable_innings
                                                                              , maximum_allowable_innings)
        home_sp_stat_df = pd.DataFrame([home_starting_pitcher_stats], columns=pitching_stat_headers)
        home_sp_stat_df = home_sp_stat_df.drop(['avg_value?', 'game_date', 'pitcher', 'num_innings'], axis=1)

        away_starting_pitcher_stats = sim_single_pitcher_prior_pitching_stats(pitching_df, date, away_pitcher_id
                                                                              , minimum_allowable_innings
                                                                              , maximum_allowable_innings)
        away_sp_stat_df = pd.DataFrame([away_starting_pitcher_stats], columns=pitching_stat_headers)
        away_sp_stat_df = away_sp_stat_df.drop(['avg_value?', 'game_date', 'pitcher', 'num_innings'], axis=1)

        full_home_lineup_fielding = calculate_lineup_stats(fielding_df, home_lu, date
                                                           , fielder_stat_agg_columns, fielder_stat_headers
                                                           , minimum_allowable_fielding_plays
                                                           , maximum_allowable_fielding_plays, 'fielder', 'game_date')

        full_away_lineup_fielding = calculate_lineup_stats(fielding_df, away_lu, date
                                                           , fielder_stat_agg_columns, fielder_stat_headers
                                                           , minimum_allowable_fielding_plays
                                                           , maximum_allowable_fielding_plays, 'fielder', 'game_date')

        print(full_home_lineup_batting)
        print(home_sp_stat_df)
        print(home_bullpen_stats)
        print(full_home_lineup_fielding)

        simulate_game(full_home_lineup_batting, full_away_lineup_batting, full_home_lineup_fielding
                      , full_away_lineup_fielding, home_sp_stat_df, away_sp_stat_df, home_bullpen_stats
                      , away_bullpen_stats, model_list, scaler_list)

        time.sleep





