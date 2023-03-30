from statsapi import schedule
import pandas as pd
from datetime import datetime


if __name__ == '__main__':

    current_day = datetime.now()

    date_string = current_day.strftime("%m/%d/%Y")

    file_date_string = current_day.strftime("%m_%d_%Y")

    games = schedule(start_date=date_string, end_date=date_string)
    result = []

    for g in games:
        game_id = g['game_id']
        home_team = g['home_name']
        away_team = g['away_name']
        row = [game_id, home_team, away_team]
        result.append(row)

    game_df = pd.DataFrame(result, columns=['game_id', 'home_team', 'away_team'])

    file_name = 'daily_games/games_' + file_date_string + '.csv'

    game_df.to_csv(file_name, index=False)


