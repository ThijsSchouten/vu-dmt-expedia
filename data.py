import pandas as pd
import numpy as np
import pickle
import random
from sklearn import preprocessing
import datetime


def read_datafile(fname):
    df = pd.read_csv(fname, nrows=10008)
    df["date_time"] = pd.to_datetime(
        df["date_time"], format="%Y-%m-%d %H:%M:%S"
    )
    df["date_time"] = df["date_time"].apply(lambda x: x.date())
    return df


def drop_columns(df):
    # Copy dataframe
    df2 = df.copy()

    # Columns that have more than 30 percent missing data:
    cols_to_drop = [
        "date_time",
        "visitor_hist_starrating",
        "visitor_hist_adr_usd",
        "srch_query_affinity_score",
        "comp1_rate",
        "comp1_inv",
        "comp1_rate_percent_diff",
        "comp2_rate_percent_diff",
        "comp3_rate_percent_diff",
        "comp4_rate_percent_diff",
        "comp5_rate_percent_diff",
        "comp6_rate_percent_diff",
        "comp7_rate_percent_diff",
        "comp8_rate_percent_diff",
        "comp2_rate",
        "comp3_rate",
        "comp4_rate",
        "comp5_rate",
        "comp6_rate",
        "comp7_rate",
        "comp8_rate",
        "comp2_inv",
        "comp3_inv",
        "comp4_inv",
        "comp5_inv",
        "comp6_inv",
        "comp7_inv",
        "comp8_inv",
        "gross_bookings_usd",
        # "srch_id",
        "prop_id",
    ]

    # Drop said columns and return new dataframe
    df2.drop(cols_to_drop, axis=1, inplace=True)

    return df2


def randomise_missing_values(df, columns_to_fill):
    df2 = df.copy()
    for col in df.columns:
        data = df2[columns_to_fill]
        mask = data.isnull()
        samples = random.choices(data[~mask].values, k=mask.sum())
        df2[columns_to_fill][mask] = samples

    return df2


def drop_and_impute(df):
    """
    Drops all columns with more than 30 percent of the data missing,
    and imputes the rest of the missing values.
    """
    df2 = df.copy()

    # Drop columns with more than 30 percent missing
    df2 = drop_columns(df2)

    # Randomise the prop_review_score column
    df2 = randomise_missing_values(df2, "prop_review_score")

    # Impute prop_location_score2 with mean
    df2["prop_location_score2"].fillna(
        (df2["prop_location_score2"].mean()), inplace=True
    )

    # Impute orig_destination_distance with median
    df2["orig_destination_distance"].fillna(
        (df2["orig_destination_distance"].median()), inplace=True
    )

    return df2


def impute_negative(df):
    """
    Imputes all missing values of a dataframe with negative values.
    """
    df2 = df.copy()

    # Impute all missing values with -1
    df2.fillna(-1, inplace=True)

    return df2


def balance_click_classes(df):
    click_indices = df[df.click_bool == 1].index
    random_indices = np.random.choice(
        click_indices, len(df.loc[df.click_bool == 1]), replace=False
    )
    click_sample = df.loc[random_indices]

    not_click = df[df.click_bool == 0].index
    random_indices = np.random.choice(
        not_click, sum(df["click_bool"]), replace=False
    )
    not_click_sample = df.loc[random_indices]

    df_new = pd.concat([not_click_sample, click_sample], axis=0)

    return df_new


def normalise_price(df):
    df2 = df.copy()

    # All the columns with respect to which we want to normalise
    ref_cols = [
        "srch_id",
        "srch_destination_id",
        "srch_booking_window",
        "prop_country_id",
        "date_time",
        "prop_id",
    ]
    # Initialise scaler

    scaler = preprocessing.MinMaxScaler()

    # Loop over all columns to reference to
    for ref_col in ref_cols:
        # Create new normalised column
        new_col = "price_" + ref_col
        df2[new_col] = np.nan

        # Loop over all unique values in reference column,
        # normalise the price, and add this to the new
        # normalised column
        for unique_val in df2[ref_col].unique():
            x = df2.loc[df2[ref_col] == unique_val, "price_usd"]
            scaled_x = scaler.fit_transform(x.values.reshape(-1, 1))
            df2.loc[df2[ref_col] == unique_val, new_col] = scaled_x

    return df2


fname = "./data/training_set_VU_DM.csv"
data = read_datafile(fname)
# data1 = drop_and_impute(data)
# data2 = impute_negative(data)
