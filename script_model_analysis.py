#%%
import pickle
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


def read_file(in_path):
    """
    Reads temp model results file.
    """
    obj = pickle.load(open(in_path, "rb"))  # %%
    return obj


def plot_feature_importance(fi, names, out_path):
    """
    Plots models feature importances
    in a sorted manner.
    """

    names = fi.keys()
    gain = fi.values()

    gain, names = zip(*sorted(zip(gain, names)))

    figure(figsize=(8, 20), dpi=150)
    plt.barh(names, gain)
    plt.xlabel("XGBoost Average Gain")

    plt.savefig(out_path)


def plot_grid_df(grid_df, feature):

    # _df = grid_df.groupby(feature).mean("score").reset_index().copy()
    # print(_df)
    _df = grid_df.copy()
    score = _df["score"]
    var = _df[feature]

    figure(figsize=(5, 3), dpi=100)
    # _df.plot.line(x=f"{feature}", y="score")
    plt.scatter(var, score)
    plt.xlabel(f"{feature}")
    plt.ylabel("Score")


#%%
ID = "V01_TRN80_GS11"
obj = read_file(f"output/results/{ID}_feature_importances.p")
fi, names, model, X_val, y_val, grid_df = obj.values()
#%%
plot_feature_importance(fi, names, f"output/results/{ID}_featureimportance.png")

#%%
cols = [x for x in grid_df.columns]
[
    cols.remove(x)
    for x in [
        "score",
        "id",
        "objective",
        "booster",
        "early_stopping_rounds",
        "eval_metric",
        "tree_method",
        "num_boost_round",
    ]
]

for col in cols:
    print(col)
    plot_grid_df(grid_df, col)


# %%

# %%
