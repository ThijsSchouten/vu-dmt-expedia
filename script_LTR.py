# %%
from lib_LTR import *
import numpy as np

ID = "V01_TRN80_GS11"
ranker = LearnToRank()

ranker.load_data(
    train_df_pickle="./data/split_set1/TRN_80_BAL.p",  # "./data/normalised_data.pickle"
    test_df_pickle="./data/normalised_test-data_2.pickle",  # "./data/split_set1/TRN_05_BAL.p"
    val_df_pickle="./data/split_set1/VAL_20_UNB.p",  # moet /data/normalised_test-data_2.pickle worden
)

ranker.add_rank(click_score=1, book_score=5)
ranker.add_qid(att="srch_id")  # query ID / group
ranker.prep_data(drop_cols=True)

# %% Run gridsearch
params = dict(
    tree_method=["hist"],
    booster=["gbtree"],
    objective=["rank:ndcg"],  # , "rank:pairwise", "rank:listwise"],
    eval_metric=["ndcg@5"],
    early_stopping_rounds=[10],
    eta=[0.1, 0.05, 0.005],
    gamma=np.arange(0, 3.5, 0.5),
    alpha=np.arange(0.7, 1, 0.15),
    min_child_weight=np.arange(0, 3, 0.5),
    reg_lambda=np.arange(0.7, 1, 0.15),
    max_depth=np.arange(4, 30, 3),
    num_boost_round=[150],
    num_parallel_tree=np.arange(1, 10, 1),
    subsample=np.arange(0.4, 1, 0.2),
    colsample_bytree=np.arange(0.4, 1, 0.2),
)
# params = dict(
#     tree_method=["hist"],
#     booster=["gbtree"],
#     objective=["rank:ndcg"],  # , "rank:pairwise", "rank:listwise"],
#     eval_metric=["ndcg@5"],
#     early_stopping_rounds=[10],
#     eta=[0.05],
#     gamma=[1.5],
#     alpha=[1.0],
#     min_child_weight=[2.0],
#     reg_lambda=[1.0],
#     max_depth=[9],
#     num_boost_round=[70],
#     num_parallel_tree=[10],
#     subsample=[0.8],
#     colsample_bytree=[0.8],
# )

ranker.gridsearch(
    params,
    out_path=f"output/results/{ID}_gridsearch.p",
    multip=False,
    n_rounds=111,
    verbose=False,
)

# ranker.load_results("output/gridsearch_0X_TEST.pickle")

#%%
params = ranker.get_best_params()

#%%
ranker.train_best_model(f"output/results/{ID}_kaggle_pred.csv")

#%%
ranker.save_model(f"output/results/{ID}_best_model.p")
ranker.save_model_meta(f"output/results/{ID}_feature_importances.p")


# def main():
#     pass

# # %%
# if __name__ == "__main__":
#     main()
