import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from general_functions import one_hot_string_encoding


def calculate_all_logistic_regressions(df, one_hot_label, x_var_labels, y_var_labels):
    unique_home_teams = np.unique(df[one_hot_label]).tolist()
    print(unique_home_teams)

    for t in unique_home_teams:
        df = one_hot_string_encoding(df, one_hot_label, t, t)
    print(df)

    x_var_labels = x_var_labels + unique_home_teams

    x_variable_df = df.filter(x_var_labels)
    x_variable_array = x_variable_df.values

    min_max_scalar = preprocessing.MinMaxScaler()

    x_train_scaler = min_max_scalar.fit_transform(x_variable_array)

    model_list = []

    for y in range(0, len(y_var_labels)):

        temp_y_label = y_var_labels[y]

        y_variable_df = df.filter([temp_y_label])
        y_variable_array = y_variable_df.values.reshape(-1, 1)
        print(y_variable_array)
        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=1
                                   , max_iter=1000, n_jobs=-1).fit(x_train_scaler, y_variable_array)
        model_list.append(model)
        predictions = model.predict_proba(x_train_scaler)

        print(model.coef_)
        # method to produce minimum complexity models??? test performance out of sample
        logistic_regression_probability_check(y_variable_df[temp_y_label], predictions[:, 1].tolist(), 70)

    result = (min_max_scalar, model_list)

    return result


def logistic_regression_probability_check(actual, predict, window):
    # print(actual)
    # print(predict)
    raw_df = pd.DataFrame(columns=['Actual', 'Prediction', 'Baseline'])
    raw_df.loc[:, 'Actual'] = actual
    raw_df.loc[:, 'Prediction'] = predict

    max_index = len(actual) - window

    temp_list = []

    for i in range(0, max_index, window):
        temp_row = []
        start_index = i
        end_index = i + window

        avg_actual = np.average(raw_df.loc[start_index:end_index, 'Actual'])
        avg_predict = np.average(raw_df.loc[start_index:end_index, 'Prediction'])

        temp_row.append(avg_actual)
        temp_row.append(avg_predict)
        temp_list.append(temp_row)

    df = pd.DataFrame(temp_list, columns=['Average Actual', 'Average Prediction'])

    df.loc[:, 'MAE'] = np.abs(df['Average Actual'] - df['Average Prediction'])
    print(df)
    print(np.average(df['MAE']))
    plt.scatter(df['Average Prediction'], df['Average Actual'])
    plt.show()

    return


if __name__ == '__main__':
    regression_data = pd.read_csv('batter_outcome_and_prior_pitching_batting_stats.csv', nrows=100000)
    regression_data = regression_data[(regression_data['avg_value?'] == 0)].reset_index(drop=True)
    regression_data = regression_data[(regression_data['average_batter?'] == 0)].reset_index(drop=True)
    regression_data = regression_data.dropna().reset_index(drop=True)

    regression_y_variable_headers = ['out_flag', 'double_play_flag', 'hit_by_pitch_flag', 'strikeout_flag', 'walk_flag'
                                     , 'single_flag', 'double_flag', 'triple_flag', 'home_run_flag']

    regression_x_variable_headers = ['era', 'whip', 'tbip', 'strikeout_rate', 'walk_rate', 'single_rate', 'double_rate'
                                     , 'triple_rate', 'home_run_rate', 'hit_by_pitch_rate', 'double_play_rate'
                                     , 'out_rate', 'batter_out_rate', 'batter_double_play_rate'
                                     , 'batter_hit_by_pitch_flag', 'batter_strikeout_rate', 'batter_walk_rate'
                                     , 'batter_single_rate', 'batter_double_rate', 'batter_triple_rate'
                                     , 'batter_home_run_rate']

    calculate_all_logistic_regressions(regression_data, 'home_team'
                                       , regression_x_variable_headers, regression_y_variable_headers)
