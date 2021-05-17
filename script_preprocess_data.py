import pickle
from lib_data import *
import random
import numpy


def create_pickle(source, target1, target2, cohort):
    data = read_datafile(source, all_data=False)
    print("CSV loaded in")

    print("Adding price_diff feature")
    # Add pricediff feature
    data = add_pricediff_feature(data)

    print("Starting normalisation")
    # Normalisation
    data = normalise_column(data, "price_usd")
    data = normalise_column(data, "prop_location_score1")
    data = normalise_column(data, "prop_location_score2")
    data = normalise_column(data, "price_diff_from_recent")

    print("Adding combinatorial features")
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

    print("Imputing missing values")
    # Missing values and class balancing
    data1 = drop_and_impute(data, cohort=cohort)
    # data = balance_click_classes(data)

    data2 = impute_negative(data)

    print("Saving files")
    # Save files
    with open(target1, "wb") as f:
        pickle.dump(data1, f)

    with open(target2, "wb") as f:
        pickle.dump(data2, f)


def main():
    random.seed(42)
    np.random.seed(42)

    # Create pickles
    create_pickle(
        "./data/training_set_VU_DM.csv",
        "./data/normalised_unbalanced_training-data.pickle",
        "./data/normalised_unbalanced_training-data_2.pickle",
        cohort="train",
    )
    create_pickle(
        "./data/test_set_VU_DM.csv",
        "./data/normalised_test-data.pickle",
        "./data/normalised_test-data_2.pickle",
        cohort="test",
    )


if __name__ == "__main__":
    main()
