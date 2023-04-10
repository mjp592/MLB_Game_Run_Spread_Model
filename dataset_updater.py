import pybaseball.cache
from pybaseball import statcast
from datetime import datetime
from datetime import timedelta
import pandas as pd


def update_main_dataset(og_dataset, current_date):

    last_data_date = og_dataset.loc[0, 'game_date'][0:10]
    last_data_date = datetime.strptime(last_data_date, '%Y-%m-%d')
    start_date = last_data_date + timedelta(1)
    start_date = start_date.strftime('%Y-%m-%d')
    print(start_date)
    df = statcast(start_dt=start_date, end_dt=current_date)
    merge_df = pd.concat([df, og_dataset]).reset_index(drop=True)
    print(og_dataset)
    print(df)
    print(merge_df)
    merge_df.to_csv('raw_dataset.csv', index=False)

    return


if __name__ == '__main__':

    today = datetime.today().strftime('%Y-%m-%d')
    og_data = pd.read_csv('raw_dataset.csv', low_memory=False)
    update_main_dataset(og_data, today)

    stream = open("core_dataset_creator.py")
    read_file = stream.read()
    exec(read_file)

    stream2 = open("fielding_dataset_generator.py")
    read_file2 = stream2.read()
    exec(read_file2)
