# %%
import pandas as pd
import matplotlib.pyplot as plt
import pickle


def describe(df, save=False):
    """
    Describes DF and optionally
    saves to file.
    """
    description = df.describe().transpose()

    if save:
        with open(save, "wb") as f:
            pickle.dump(description, f)

    return description


def plot_histograms(df, path, dpi=300):
    """
    Generates a histogram for each df column
    and save it to specified path.
    """
    for var in df.columns:
        if var == "date_time" or var == "srch_id":
            continue
        try:
            print(var)
            series = df[var]
            plt.clf()
            plt.hist(series, bins=100)
            plt.title(f"{var}")
            plt.savefig(path + f"{var}.png", dpi=dpi)

        except:
            pass


def plot_missing_values(df, path, dpi=300):
    """
    Computes percentage of values missing
    and plots as barchart to specified path.
    """
    values_array = []
    for var in sorted(df.columns):
        series = df[var]
        perc_missing = 100 - series.count() / len(series) * 100
        values_array.append({"var": var, "% missing": perc_missing})

    df = pd.DataFrame(values_array)
    ax = df.plot.bar(x="var", y="% missing", figsize=(15, 4))
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")

    pass


def plot_boxplot(_df, path, drop=[], dpi=300):
    """

    """
    df = _df.copy()
    for col in drop:
        del df[col]

    ax = df.boxplot(figsize=(15, 6), rot=90)
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
