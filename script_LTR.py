# %%
from lib_LTR import *
import pickle

ID = "A1"

# %%
# Initialize ranker
ranker = LearnToRank()

ranker.load_data(
    train_df_pickle="./data/split/train80prc.pickle",  # "./data/normalised_data.pickle"
    test_df_pickle="./data/normalised_test-data.pickle",
    val_df_pickle="./data/split/val20prc.pickle",
)

ranker.add_rank(click_score=1, book_score=5)
ranker.add_qid(att="srch_id")  # query ID / group
ranker.prep_data(drop_cols=True)
# ranker.create_groupsplits(test_size=0.25, splits=2)

# %% Run gridsearch
params = dict(
    tree_method=["hist"],
    booster=["gbtree"],
    objective=["rank:ndcg"],
    # random_state=[42],
    colsample_bytree=[1, 0.9, 0.5],
    eta=[0.1, 0.05, 0.005],
    max_depth=[5, 7, 9, 12, 15],
    n_estimators=[60, 35, 20],
    subsample=[0.9, 0.6, 0.4],
)
params = dict(
    compute_importances=[True],
    tree_method=["hist"],
    booster=["gbtree"],
    objective=["rank:pairwise"],
    # random_state=[42],
    colsample_bytree=[0.9],
    eta=[0.1],
    max_depth=[9],
    n_estimators=[150],
    subsample=[0.9],
)

ranker.gridsearch(params, out_path=f"output/results/{ID}_gridsearch.pickle")
gridsearch_results = ranker.gridsearch_results

# %%
# ranker.load_results("output/gridsearch_0X_TEST.pickle")
ranker.get_best_params()
ranker.train_best_model(f"output/results/{ID}_kaggle_pred.csv")
ranker.best_model.save_model(f"output/results/{ID}_best_model.model")

#%% Write files to pickle for later analysis
fi = ranker.best_model.feature_importances_
names = ranker.X_trn.columns
bm = ranker.best_model
X_val = ranker.X_val
y_val = ranker.y_val
out_obj = dict(fi=fi, names=names, model=bm, X_val=X_val, y_val=y_val)

fname = f"output/results/{ID}_feature_importances.p"
pickle.dump(out_obj, open(fname, "wb"))

# feature_ranker.best_model.feature_importances_

# %%
