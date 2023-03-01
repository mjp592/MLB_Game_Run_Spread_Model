import pandas as pd
from pybaseball import team_game_logs
from pybaseball import statcast_single_game
from statsapi import schedule
from statsapi import lookup_player
from statsapi import roster
from pybaseball.lahman import appearances
import requests
import json


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


if __name__ == '__main__':

    games = schedule(start_date='07/01/2018', end_date='07/31/2018')
    print(appearances())
    for x in games:
        temp_game_id = x['game_id']
        home_lu, away_lu, home_pitcher, away_pitcher, date, home_id, away_id = get_lineups_for_game(temp_game_id, 2018)
        print(roster(home_id, date=date))
        time.sleep





