import pandas as pd
import numpy as np
import pickle
import random
from statsmodels.regression import linear_model
from sklearn.preprocessing import PolynomialFeatures
from contextlib import redirect_stdout

def read_save_datafile(fname):
    df = pd.read_csv(fname, index_col=0)
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
        data[mask] = samples

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
    with open("output/modelsummary.txt", "w+") as f:
        print(s, file=f)


def interaction_effects(data,dep_variable, threshold):
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
    with open("output/modelsummary.csv", "w") as f:
        f.write(model.summary().as_csv())
    # show significant results
    results = pd.read_csv("output/modelsummary.csv", skiprows= 10, skipfooter= 10, index_col= 0)
    results.index = names
    sign_results = results[results["P>|t| "] < threshold]
    print(sign_results.sort_values(by = '   coef   ', ascending = False))

fname = "./data/training_set_VU_DM.csv"
# data = read_save_datafile(fname)
