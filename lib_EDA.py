# %%
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np


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
    """"""
    df = _df.copy()
    for col in drop:
        del df[col]

    ax = df.boxplot(figsize=(15, 6), rot=90)
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")


def create_price_variance_df(
    df_path="data/training_set_VU_DM.csv",
    out_path="output/EDA_price_var2.pickle",
):
    ref_cols = [
        "srch_id",
        "srch_booking_window",
        "prop_country_id",
        "date_time",
        "prop_id",
    ]
    df = pd.read_csv(df_path, usecols=(ref_cols + ["price_usd"]), nrows=100)

    plot_df = pd.DataFrame(columns=ref_cols)
    # Convert date time to monthsi
    df["date_time"] = pd.to_datetime(
        df["date_time"], format="%Y-%m-%d %H:%M:%S"
    )
    df["date_time"] = df["date_time"].apply(lambda x: (x.year, x.month))

    # Remove outliers somewhat?
    df = df.loc[df["price_usd"] < 0.75e-7]

    for ref_col in ref_cols:
        means = []
        for unique_val in df[ref_col].unique():
            mean = df.loc[df[ref_col] == unique_val, "price_usd"].mean()
            means.append(mean)
        # print(ref_col, means)
        # print(np.std(means))
        plot_df[ref_col] = [np.std(np.array(means))]

    # with open(out_path, "wb") as f:
    #     pickle.dump(plot_df, f)
    return df, plot_df


def plot_price_var(
    pickle_path="output/EDA_price_var.pickle",
    output_path="output/EDA_price_var.png",
    dpi=300,
):
    with open(pickle_path, "rb") as f:
        df = pickle.load(f)

    stds = np.sqrt(df.to_numpy().squeeze())
    data = {"Feature": list(df.columns), "Standard Deviation": stds}
    df2 = pd.DataFrame(data)
    ax = df2.plot.bar(x="Feature", y="Standard Deviation", figsize=(8, 4))
    ax.figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
