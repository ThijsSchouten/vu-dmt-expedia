from data import *

data = read_save_datafile("data/training_set_VU_DM.csv")
data = data.sample(frac=0.5)
data = drop_and_impute(data)

interaction_effects(data)
