import pybaseball.cache
from pybaseball import statcast
from datetime import datetime


def download_raw_pitch_data(filename, start_date, end_date, append_flag):
    pybaseball.cache.enable()

    df = statcast(start_dt=start_date, end_dt=end_date)
    print(df)

    if append_flag == 1:
        df.to_csv(filename, index=False, mode='a', header=False, )
    else:
        df.to_csv(filename, index=False)

    return df


if __name__ == '__main__':

    start = '2008-01-01'
    today = datetime.today().strftime('%Y-%m-%d')

    download_raw_pitch_data('raw_dataset.csv', start, today, 0)
