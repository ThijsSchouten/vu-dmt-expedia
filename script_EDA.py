#%%
from lib_EDA import *

training_set = pd.read_csv("data/training_set_VU_DM.csv")
training_set["date_time"] = pd.to_datetime(training_set["date_time"])

# %%
describe(training_set, "output/EDA_descriptive_df.pickle")

# %%
plot_histograms(training_set, path="output/EDA_histograms/")
# %%
plot_missing_values(training_set, path="output/EDA_missing_values.png")
# %%
plot_boxplot(
    training_set,
    path="output/EDA_boxplot.png",
    drop=["prop_id", "srch_destination_id", "orig_destination_distance"],
)
