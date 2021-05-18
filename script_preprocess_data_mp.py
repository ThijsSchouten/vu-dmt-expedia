import pickle
from lib_data import *
from script_add_features import *
import random
import numpy
import multiprocessing as mp

import warnings
import time


def create_pickle(source, target, cohort, imputation="standard"):
    print(
        f"\n  Source: {source}\n  Target: {target}\n  Cohort:{cohort}\n  Imputation:{imputation}\n{'-'*70}"
    )
    data = read_datafile(source, all_data=False, nrows=1000)
    print("CSV loaded")

    print("Adding price_diff feature")
    # Add pricediff feature
    data = add_pricediff_feature(data)

    print("Starting normalisation")
    # Normalisation
    data = normalise_column(data, "price_usd")
    data = normalise_column(data, "prop_location_score1")
    data = normalise_column(data, "prop_location_score2")
    data = normalise_column(data, "price_diff_from_recent")

    print("Adding combinatorial and other features")
    # Define features to create combinatorial collumns from
    comb_feats = [
        ["orig_destination_distance", "promotion_flag"],
        ["srch_length_of_stay", "promotion_flag"],
        ["prop_location_score2", "promotion_flag"],
        ["prop_location_score1", "prop_location_score2"],
    ]

    # Loop over pairs and create columns
    for f1, f2 in comb_feats:
        add_combination_feature(data, f1, f2, inplace=True)

    data = create_checkin_checkout(data)
    data = create_price_ranks(data)

    print("Imputing missing values")

    # Missing values and class balancing
    if imputation == "standard":
        data = drop_and_impute(data, cohort=cohort)

    if imputation == "negative":
        data = impute_negative(data)

    print(f"Saving file to {target}\n")
    # Save files
    with open(target, "wb") as f:
        pickle.dump(data, f)


def normalise_remainder(pname):
    with open(pname, "rb") as f:
        data = pickle.load(f)

    data = normalise_column(data, "prop_location_score1")
    data = normalise_column(data, "prop_location_score2")

    with open(pname, "wb") as f:
        pickle.dump(data, f)


def main():
    start = time.time()

    random.seed(42)
    np.random.seed(42)

    pool = mp.Pool(200)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        # Create pickles
        args = [
            [
                "./data/training_set_VU_DM.csv",
                "./data/normalised_unbalanced_training-data_2.pickle",
                "train",
                "negative",
            ],
            [
                "./data/test_set_VU_DM.csv",
                "./data/normalised_test-data_2.pickle",
                "test",
                "negative",
            ],
            [
                "./data/training_set_VU_DM.csv",
                "./data/normalised_unbalanced_training-data.pickle",
                "train",
                "standard",
            ],
            [
                "./data/test_set_VU_DM.csv",
                "./data/normalised_test-data.pickle",
                "test",
                "standard",
            ],
        ]

        results = pool.starmap(create_pickle, args)

        # normalise_remainder("./data/normalised_unbalanced_training-data.pickle")
        # normalise_remainder("./data/normalised_test-data.pickle")

    print(time.time() - start)


if __name__ == "__main__":
    main()
