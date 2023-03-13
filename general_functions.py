import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error


def one_hot_string_encoding(df, existing_label, matching_string, new_label):
    df.loc[df[existing_label] == matching_string, new_label] = 1
    df.loc[df[existing_label] != matching_string, new_label] = 0

    return df


def transform_and_remove_unwanted_events(df, label, remove_iterable, remap_iterable, new_value_iterable):

    if len(remap_iterable) != len(new_value_iterable):
        print('Each remapping value does not have a direct mapping to a replacement value.')
        return df

    for k in remove_iterable:

        df = df[(df[label] != k)]

    df = df.reset_index(drop=True)

    for i in range(0, len(remap_iterable)):
        old_value = remap_iterable[i]
        new_value = new_value_iterable[i]
        df.loc[df[label] == old_value, label] = new_value

    return df


def k_fold_generation(features, label, k):

    if features.shape[0] == label.shape[0]:
        number_of_observations = features.shape[0]
    else:
        return "Datasets do not match in length."

    datasets = {}

    dataset_divisor = int(number_of_observations / k)

    for i in range(0, k):
        key_feature = "feature" + str(i)
        key_label = "label" + str(i)

        start_index = i * dataset_divisor

        if i != (k-1):
            end_index = (i + 1) * dataset_divisor
        else:
            end_index = number_of_observations - 1

        temp_feature = features[start_index:end_index, :]
        temp_label = label[start_index:end_index]

        datasets[key_feature] = temp_feature
        datasets[key_label] = temp_label

    return datasets


def forward_chain_dataset_build(datasets, k):
    training_feature_array = datasets['feature0']
    training_label_array = datasets['label0']

    for i in range(1, k):

        test_feature = "feature" + str(i)
        test_label = "label" + str(i)

        test_feature_array = datasets[test_feature]
        test_label_array = datasets[test_label]

        if i != 1 or i != (k - 1):
            training_feature = "feature" + str(i - 1)
            training_label = "label" + str(i - 1)

            temp_feature = datasets[training_feature]
            temp_label = datasets[training_label]

            training_feature_array = np.concatenate((training_feature_array, temp_feature), axis=0)
            training_label_array = np.concatenate((training_label_array, temp_label), axis=0)

            del temp_feature
            del temp_label

    result_tup = (training_feature_array, training_label_array, test_feature_array, test_label_array)

    return result_tup


# General Statistics Testing Modules -------------------------------------------------------------------------------
def forward_chain_train_logistic_regression_model(train_features, train_labels, test_features, test_labels, model
                                                  , feature_labels, standard_flag, label_scaler):

    model.fit(train_features, train_labels)
    predictions = model.predict_proba(test_features)
    predictions = predictions[:, 1].reshape(-1, 1)

    if standard_flag:
        predictions = label_scaler.inverse_transform(predictions)

    input_output_df = pd.DataFrame(test_features, columns=feature_labels)
    input_output_df.loc[:, 'Actual Value'] = test_labels
    input_output_df.loc[:, 'Predicted Value'] = predictions
    # input_output_df.loc[:, 'Residuals'] = input_output_df['Actual Value'] - input_output_df['Predicted Value']

    return input_output_df

