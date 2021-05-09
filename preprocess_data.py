import pickle
from data import *


def main():

    data = read_datafile("./data/training_set_VU_DM.csv", all_data=False)
    data = normalise_price(data)
    data = drop_and_impute(data)
    data = balance_click_classes(data)
    with open("./data/normalised_data.pickle", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
