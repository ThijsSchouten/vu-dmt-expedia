# %%
import pandas as pd
import random
import math


def main():
    balanced_set = pd.read_pickle("data/normalised_data.pickle")
    unbalanced_set = pd.read_pickle("data/normalised_unbalanced_training-data.pickle")

    # Get unique IDs, count how many there are. Then shuffle
    qids = balanced_set.srch_id.unique()
    count = len(qids)
    random.shuffle(qids)

    # Split into 80 train / 20 val
    train_indices = qids[: math.floor(count * 0.8)]
    val_indices = qids[math.floor(count * 0.8) :]

    # train_80prc_unbalanced = unbalanced_set[unbalanced_set["srch_id"].isin(train_indices)]
    train_20prc_unbalanced = unbalanced_set[unbalanced_set["srch_id"].isin(val_indices)]

    train_80prc_balanced = balanced_set[balanced_set["srch_id"].isin(train_indices)]
    # train_20prc_balanced = balanced_set[balanced_set["srch_id"].isin(val_indices)]

    train_80prc_balanced.to_pickle("data/split/train80prc.pickle")
    train_20prc_unbalanced.to_pickle("data/split/val20prc.pickle")


if __name__ == "__main__":
    main()


# %%
