# %%
import pandas as pd
import random
import math
from pathlib import Path

from lib_data import balance_click_classes

#%%
def main():
    OUT_DIR = "data/split_set1"
    DATA_IN = "data/normalised_unbalanced_training-data_2.pickle"

    # Create path if needed
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load dataset
    unbalanced_set = pd.read_pickle(DATA_IN)

    # %% Get unique IDs, count how many there are. Then shuffle
    qids = unbalanced_set.srch_id.unique()
    random.shuffle(qids)
    count = len(qids)

    # %% Generate list of QIDs for specified splits
    train_indices_80prc = qids[: math.floor(count * 0.8)]
    train_indices_20prc = qids[: math.floor(count * 0.20)]
    train_indices_05prc = qids[: math.floor(count * 0.05)]
    val_indices_20prc = qids[math.floor(count * 0.8) :]

    # %% Split DF on qid and create a balanced set
    # for the training variations.
    df_train_80_UNB = unbalanced_set[
        unbalanced_set["srch_id"].isin(train_indices_80prc)
    ]
    df_train_80_BAL = balance_click_classes(df_train_80_UNB)

    df_train_20_UNB = unbalanced_set[
        unbalanced_set["srch_id"].isin(train_indices_20prc)
    ]
    df_train_20_BAL = balance_click_classes(df_train_20_UNB)

    df_train_05_UNB = unbalanced_set[
        unbalanced_set["srch_id"].isin(train_indices_05prc)
    ]
    df_train_05_BAL = balance_click_classes(df_train_05_UNB)

    df_val_20_UNB = unbalanced_set[unbalanced_set["srch_id"].isin(val_indices_20prc)]

    # %% Print sizes
    print("unb train 80%", df_train_80_UNB.shape)
    print("bal train 80%", df_train_80_BAL.shape)

    print("unb train 20%", df_train_20_UNB.shape)
    print("bal train 20%", df_train_20_BAL.shape)

    print("unb train 05%", df_train_05_UNB.shape)
    print("bal train 05%", df_train_05_BAL.shape)

    print("unb val 20%", df_val_20_UNB.shape)

    # %% Write to pickles
    df_train_80_UNB.to_pickle(fr"{OUT_DIR}/TRN_80_UNB.p")
    df_train_80_BAL.to_pickle(fr"{OUT_DIR}/TRN_80_BAL.p")

    df_train_20_UNB.to_pickle(fr"{OUT_DIR}/TRN_20_UNB.p")
    df_train_20_BAL.to_pickle(fr"{OUT_DIR}/TRN_20_BAL.p")

    df_train_05_UNB.to_pickle(fr"{OUT_DIR}/TRN_05_UNB.p")
    df_train_05_BAL.to_pickle(fr"{OUT_DIR}/TRN_05_BAL.p")

    df_val_20_UNB.to_pickle(fr"{OUT_DIR}/VAL_20_UNB.p")

    print("Done.")


if __name__ == "__main__":
    main()


# %%
