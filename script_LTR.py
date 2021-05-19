# %%
from lib_LTR import *
import numpy as np

ID = "V02_TRN80_GS1"
ranker = LearnToRank()

ranker.load_data(
    train_df_pickle="./data/split_set1/TRN_80_UNB.p",  # "./data/normalised_data.pickle"
    test_df_pickle="./data/normalised_test-data_2.pickle",  # "./data/split_set1/TRN_05_BAL.p"
    val_df_pickle="./data/split_set1/VAL_20_UNB.p",  # moet /data/normalised_test-data_2.pickle worden
)

ranker.add_rank(click_score=1, book_score=5)
ranker.add_qid(att="srch_id")  # query ID / group
ranker.prep_data(drop_cols=True)

# %% Run gridsearch
# params = dict(
#     tree_method=["hist"],
#     booster=["gbtree"],
#     objective=["rank:ndcg"],  # , "rank:pairwise", "rank:listwise"],
#     eval_metric=["ndcg@5"],
#     early_stopping_rounds=[8],
#     eta=[0.1, 0.05, 0.005],
#     gamma=np.arange(0, 4, 0.5),
#     alpha=np.arange(0.7, 1, 0.15),
#     min_child_weight=np.arange(0, 3, 0.5),
#     reg_lambda=np.arange(0.7, 1, 0.15),
#     max_depth=np.arange(4, 30, 3),
#     num_boost_round=[100],
#     num_parallel_tree=np.arange(5, 10, 1),
#     subsample=np.arange(0.5, 1, 0.25),
#     colsample_bytree=np.arange(0.5, 1, 0.25),
# )
# .381..
params = dict(
    tree_method=["hist"],
    booster=["gbtree"],
    objective=["rank:ndcg"],  # , "rank:pairwise", "rank:listwise"],
    eval_metric=["ndcg@5"],
    early_stopping_rounds=[7],
    eta=[0.1],
    gamma=[2.5],
    alpha=[0.7],
    min_child_weight=[2.5],
    reg_lambda=[0.85],
    max_depth=[16],
    num_boost_round=[60],
    num_parallel_tree=[1],
    subsample=[0.75],
    colsample_bytree=[0.5],
)


# # .381..
# params = dict(
#     tree_method=["hist"],
#     booster=["gbtree"],
#     objective=["rank:ndcg"],  # , "rank:pairwise", "rank:listwise"],
#     eval_metric=["ndcg@5"],
#     early_stopping_rounds=[7],
#     eta=[0.1],
#     gamma=[2.5],
#     alpha=[0.7],
#     min_child_weight=[2.5],
#     reg_lambda=[0.85],
#     max_depth=[16],
#     num_boost_round=[60],
#     num_parallel_tree=[6],
#     subsample=[0.75],
#     colsample_bytree=[0.5],
# )

ranker.gridsearch(
    params,
    out_path=f"output/results/{ID}_gridsearch.p",
    multip=False,
    n_rounds=200,
    verbose=True,
)

# ranker.load_results("output/gridsearch_0X_TEST.pickle")

#%%
params = ranker.get_best_params()
ranker.train_best_model(f"output/results/{ID}_kaggle_pred.csv")

#%%
ranker.save_model(f"output/results/{ID}_best_model.p")
ranker.save_model_meta(f"output/results/{ID}_feature_importances.p")
