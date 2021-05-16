import pandas as pd
import numpy as np
import pickle
import random
from sklearn import preprocessing
import datetime
from statsmodels.regression import linear_model
from sklearn.preprocessing import PolynomialFeatures
from contextlib import redirect_stdout


def read_datafile(fname, all_data=False, nrows=10008):
    if all_data:
        df = pd.read_csv(fname)
    else:
        df = pd.read_csv(fname, nrows=nrows)
    df["date_time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d %H:%M:%S")
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
        "srch_id",
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
        df2.loc[mask, columns_to_fill] = samples

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
    random_indices = np.random.choice(not_click, sum(df["click_bool"]), replace=False)
    not_click_sample = df.loc[random_indices]

    df_new = pd.concat([not_click_sample, click_sample], axis=0)

    return df_new


def normalise_price(df, feature, average=False):
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
        new_col = feature + "_" + ref_col
        df2[new_col] = np.nan

        # Loop over all unique values in reference column,
        # normalise the price, and add this to the new
        # normalised column
        for unique_val in df2[ref_col].unique():
            x = df2.loc[df2[ref_col] == unique_val, feature]
            scaled_x = scaler.fit_transform(x.values.reshape(-1, 1))
            df2.loc[df2[ref_col] == unique_val, new_col] = scaled_x

    if average:
        df2[feature] = df2[ref_cols].mean(axis=1)
    else:
        df2.drop([feature], axis=1, inplace=True)

    return df2


def PolynomialFeatureNames(sklearn_feature_name_output, df):
    """
    This function takes the output from the .get_feature_names() method on the PolynomialFeatures
    instance and replaces values with df column names to return output such as 'Col_1 x Col_2'

    sklearn_feature_name_output: The list object returned when calling .get_feature_names() on the PolynomialFeatures object
    df: Pandas dataframe with correct column names
    """

    import re

    cols = df.columns.tolist()
    feat_map = {"x" + str(num): cat for num, cat in enumerate(cols)}
    feat_string = ",".join(sklearn_feature_name_output)
    for k, v in feat_map.items():
        feat_string = re.sub(fr"\b{k}\b", v, feat_string)
    return feat_string.replace(" ", " x ").split(",")


def myprint(s):
    with open("output/interaction_effects/modelsummary.txt", "w+") as f:
        print(s, file=f)


def interaction_effects(data, dep_variable, threshold):
    # remove date variable and booking/click depending on dep_variable
    data = data.iloc[:, 1:].copy()
    if dep_variable == "click_bool":
        data = data.drop("booking_bool", axis=1)
    elif dep_variable == "booking_bool":
        data = data.drop("click_bool", axis=1)

    # choose dependent variable
    X = data.drop(dep_variable, axis=1)
    y = data[dep_variable]
    # Generate interaction terms
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    X_interaction = poly.fit_transform(X)
    # Get names of these terms
    names = PolynomialFeatureNames(poly.get_feature_names(), X)
    # Fit model to check importance of features
    model = linear_model.OLS(y, X_interaction).fit()
    # save results
    with open("output/interaction_effects/modelsummary.csv", "w") as f:
        f.write(model.summary().as_csv())
    # show significant results
    results = pd.read_csv(
        "output/interaction_effects/modelsummary.csv",
        skiprows=10,
        skipfooter=10,
        index_col=0,
    )
    results.index = names
    sign_results = results[results["P>|t| "] < threshold]
    print(sign_results.sort_values(by="   coef   ", ascending=False))


fname = "./data/training_set_VU_DM.csv"
data = read_datafile(fname)
# data1 = drop_and_impute(data)
# data2 = impute_negative(data)
