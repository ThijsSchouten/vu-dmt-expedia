# %%
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.frame import Series

import matplotlib.pyplot as plt

from math import ceil

# %%
train_set = pd.read_csv("data/training_set_VU_DM.csv", nrows=100000)
# test_set = pd.read_csv("data/test_set_VU_DM.csv")
# submission_sample = pd.read_csv("data/submission_sample.csv")

# %%
def variable_histograms(df, output_path, size=(60, 80)):

    variables = df.columns
    variable_count = len(variables)

    cols, rows = 4, ceil(variable_count / 4)
    fig, axs = plt.subplots(rows, cols, figsize=size)
    axs = axs.ravel()
    # print(fig, axs)

    for idx, var in enumerate(variables):
        try:
            series = df[var]
            axs[idx].hist(series, bins=100)
            axs[idx].set_title(f"{var}")
        except:
            pass

    fig.savefig(output_path, dpi=300, transparent=False)


# %%
variable_histograms(train_set, "plots/eda_histograms_100000.png")


# %%
