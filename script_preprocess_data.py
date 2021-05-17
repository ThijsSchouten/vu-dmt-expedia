import pickle
from lib_data import *
import random
import numpy


def main():
    random.seed(42)
    np.random.seed(42)
    data = read_datafile("./data/training_set_VU_DM.csv", all_data=False)
    data = normalise_column(data, "price_usd")
    data = normalise_column(data, "prop_location_score1")
    data = normalise_column(data, "prop_location_score2")
    data = drop_and_impute(data)
    data = balance_click_classes(data)

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

    # Save file
    with open("./data/normalised_data.pickle", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
