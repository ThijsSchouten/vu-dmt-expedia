from lib_LTR import *

# Initialize ranker
ranker = LearnToRank()

ranker.load_data(
    # "./data/normalised_unbalanced_training-data.pickle",
    "./data/normalised_data.pickle",
    "./data/normalised_test-data.pickle",
)

ranker.add_rank()
ranker.add_qid(att="srch_id")  # query ID / group

# Then, prepare the data. Make
# sure train&test columns match
ranker.prep_data(drop_cols=True)

# %% Split the data by groups and run gridsearch
ranker.create_groupsplits(test_size=0.25, splits=2)

# Run gridsearch
fit_params = dict(
    tree_method="hist", booster="gbtree", objective="rank:ndcg", random_state=42,
)

grid_params = dict(
    learning_rate=[0.1, 0.05, 0.005],
    colsample_bytree=[1, 0.9, 0.5],
    eta=[0.05, 0.005],
    max_depth=[9, 12, 15],
    n_estimators=[90, 60, 30],
    subsample=[0.9, 0.6, 0.4],
)
# grid_params = dict(max_depth=[7], n_estimators=[7])

ranker.gridsearch(
    fit_params, grid_params, out_path="output/gridsearch_02_unbalanced.pickle"
)
grouped = ranker.get_best_params()

# %%
ranker.load_results("output/gridsearch_02_unbalanced.pickle")
ranker.get_best_params()
ranker.train_best_model("output/resultsY.csv")

# %% ToDo: analyze feature importances
res = ranker.results_aggregated
