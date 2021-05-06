from data import *
import pandas as pd

data = read_save_datafile("data/training_set_VU_DM.csv")
data = data.sample(frac=0.3)
data = drop_and_impute(data)
interaction_effects(data,"click_bool", 0.05)

