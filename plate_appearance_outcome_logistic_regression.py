import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from general_functions import one_hot_string_encoding
from general_functions import k_fold_generation
from general_functions import forward_chain_dataset_build
from general_functions import forward_chain_train_logistic_regression_model


def calculate_all_logistic_regressions(df, one_hot_label, x_var_labels, y_var_labels, num_of_folds):
    unique_home_teams = np.unique(df[one_hot_label]).tolist()

    for t in unique_home_teams:
        df = one_hot_string_encoding(df, one_hot_label, t, t)
    print(df)

    x_var_labels = x_var_labels + unique_home_teams

    x_variable_df = df.filter(x_var_labels)
    x_variable_array = x_variable_df.values

    scaler_list = []
    model_list = []
    summary_df_list = []
    sampling_list = []

    for y in range(0, len(y_var_labels)):

        temp_y_label = y_var_labels[y]

        y_variable_df = df.filter([temp_y_label])
        # y_variable_array = y_variable_df.values.reshape(-1, 1)
        y_variable_array = np.ravel(y_variable_df.values)
        folded_data = k_fold_generation(x_variable_array, y_variable_array, num_of_folds)

        for k in range(num_of_folds, num_of_folds + 1):

            training_features, training_labels, test_features, test_labels = forward_chain_dataset_build(folded_data, k)

            min_max_scalar = preprocessing.MinMaxScaler()
            scaled_training_features = min_max_scalar.fit_transform(training_features)
            scaled_test_features = min_max_scalar.transform(test_features)

            model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=1
                                       , max_iter=1000, n_jobs=-1)

            summary_df = forward_chain_train_logistic_regression_model(scaled_training_features
                                                                       , training_labels, scaled_test_features
                                                                       , test_labels, model, x_var_labels
                                                                       , False, 'NA')
            # print(summary_df)
            sampling_df = logistic_regression_probability_check(summary_df['Actual Value']
                                                                , summary_df['Predicted Value'], 70)
            # print(sampling_df)
            if num_of_folds == k:
                summary_df_list.append(summary_df)
                sampling_list.append(sampling_df)

        min_max_scalar = preprocessing.MinMaxScaler()
        scaled_features = min_max_scalar.fit_transform(x_variable_array)

        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=1
                                   , max_iter=1000, n_jobs=-1)

        model = model.fit(scaled_features, y_variable_array)

        scaler_list.append(min_max_scalar)
        model_list.append(model)

    result = (scaler_list, model_list, summary_df_list, sampling_list)

    return result


def logistic_regression_probability_check(actual, predict, window):

    raw_df = pd.DataFrame(columns=['Actual', 'Prediction'])
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

    df.loc[:, 'Residual'] = df['Average Actual'] - df['Average Prediction']
    df.loc[:, 'MAE'] = np.abs(df['Residual'])
    df.loc[:, 'MSE'] = (df['Residual'])**2
    print(np.average(df['MAE']))
    print(np.average(df['MSE']))
    # plt.hist(df['Residual'], bins=20)
    # plt.show()
    # plt.scatter(df['Average Prediction'], df['Average Actual'])
    # plt.show()

    return df


def export_csv(df, variable, file_ending):
    filename = variable + file_ending
    df.to_csv(filename, index=False)

    return


def export_joblib(obj, variable, file_ending):
    filename = variable + file_ending
    joblib.dump(obj, filename)

    return


def result_export(iterable, variables, file_ending, joblib_flag):

    for i in range(0, len(iterable)):
        temp_file = iterable[i]
        temp_var = variables[i]

        if joblib_flag:
            export_joblib(temp_file, temp_var, file_ending)
        else:
            export_csv(temp_file, temp_var, file_ending)

    return


if __name__ == '__main__':
    regression_data = pd.read_csv('batter_outcome_and_prior_pitching_batting_stats.csv')
    print(regression_data)
    regression_data = regression_data[(regression_data['avg_value?'] == 0)].reset_index(drop=True)
    regression_data = regression_data[(regression_data['average_batter?'] == 0)].reset_index(drop=True)
    regression_data = regression_data.dropna().reset_index(drop=True)
    regression_data = regression_data.iloc[::-1].reset_index(drop=True)

    regression_y_variable_headers = ['out_flag', 'double_play_flag', 'hit_by_pitch_flag', 'strikeout_flag', 'walk_flag'
                                     , 'single_flag', 'double_flag', 'triple_flag', 'home_run_flag']

    regression_x_variable_headers = ['era', 'whip', 'tbip', 'strikeout_rate', 'walk_rate', 'single_rate', 'double_rate'
                                     , 'triple_rate', 'home_run_rate', 'hit_by_pitch_rate', 'double_play_rate'
                                     , 'out_rate', 'batter_out_rate', 'batter_double_play_rate'
                                     , 'batter_hit_by_pitch_flag', 'batter_strikeout_rate', 'batter_walk_rate'
                                     , 'batter_single_rate', 'batter_double_rate', 'batter_triple_rate'
                                     , 'batter_home_run_rate']

    min_max_list, regress_list, summary_list, sample_list = calculate_all_logistic_regressions(regression_data
                                                                                               , 'home_team'
                                                                                               ,
                                                                                               regression_x_variable_headers,
                                                                                               regression_y_variable_headers,
                                                                                               5)

    output_var_list = ['out', 'double_play', 'hit_by_pitch', 'strikeout', 'walk', 'single'
                       , 'double', 'triple', 'home_run']

    result_export(min_max_list, output_var_list, '_min_max_scaler.joblib', True)
    result_export(regress_list, output_var_list, '_logistic_regression_model.joblib', True)
    result_export(summary_list, output_var_list, '_result_summary.csv', False)
    result_export(sample_list, output_var_list, '_result_sampling.csv', False)
