#%%
import xgboost as xgb
import pickle
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(10, 6), dpi=150)


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
    print(fi)
    sorted_idx = fi.argsort()
    plt.barh(names[sorted_idx], fi[sorted_idx])
    plt.xlabel("Xgboost Feature Importance")

    plt.savefig(out_path)


def main():
    ID = "A1"
    obj = read_file(f"output/results/{ID}_feature_importances.p")
    fi, names, model, X_val, y_val = obj.values()

    plot_feature_importance(fi, names, f"output/results/{ID}_featureimportance.png")


# %%
if __name__ == "__main__":
    main()

# %%
