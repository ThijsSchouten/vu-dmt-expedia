from data import *


data = read_save_datafile("data/training_set_VU_DM.csv")
data = drop_and_impute(data)
interaction_effects(data)