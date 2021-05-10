#%%
import pickle
import xgboost as xgb

#%%
with open("./data/normalised_data.pickle", "rb") as f:
    train_df = pickle.load(f)

#%%
dtrain = xgb.DMatrix(train_df)
param = {"max_depth": 2, "eta": 1, "objective": "rank:ndcg"}
num_round = 2

bst = xgb.train(param, dtrain, num_round)

# %%
