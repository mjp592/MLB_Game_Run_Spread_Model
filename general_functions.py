import pandas as pd
import numpy as np


def one_hot_string_encoding(df, existing_label, matching_string, new_label):
    df.loc[df[existing_label] == matching_string, new_label] = 1
    df.loc[df[existing_label] != matching_string, new_label] = 0

    return df
