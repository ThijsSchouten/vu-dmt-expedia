import pickle
from lib_data import *
import random
import numpy


def main():
    random.seed(42)
    np.random.seed(42)
    data = read_datafile("./data/training_set_VU_DM.csv", all_data=False)
    data = normalise_column(data, "price_usd")
    data = drop_and_impute(data)
    data = balance_click_classes(data)
    with open("./data/normalised_data.pickle", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
