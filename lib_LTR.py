#%%
import itertools as it
import multiprocessing as mp

import pickle
import random
import time

import pandas as pd
import xgboost as xgb

# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import ndcg_score

from lib_ndcg import custom_ndcg_score_groups
from lib_data import *


class LearnToRank:
    custom_validation = False

    def load_data(self, train_df_pickle, test_df_pickle, val_df_pickle):
        """
        Loads data from picklefiles.
        """
        self.train = pd.read_pickle(train_df_pickle)
        self.test = pd.read_pickle(test_df_pickle)
        self.val = pd.read_pickle(val_df_pickle)

    def load_results(self, results_df_pickle):
        """
        Loads gridsearch results from picklefiles.
        Only necessary 
        """
        print("Loaded results")
        self.results = pd.read_pickle(results_df_pickle)

    def drop_redundant_cols(self):
        """
        Drops columns from training data
        if not in testing data.
        """
        print("Dropping redundant columns.")
        test_columns = self.X_tst.columns
        train_columns = self.X_trn.columns
        val_columns = self.X_val.columns

        cols_to_drop = [x for x in train_columns if x not in test_columns]
        cols_to_drop.append("date_time")
        self.X_tst.drop(columns=["date_time"], inplace=True)

        self.X_trn.drop(columns=cols_to_drop, inplace=True)
        print(f"  Dropped columns {cols_to_drop} from X_train.")

        add_drop = [x for x in val_columns if x not in train_columns]
        all_drop = cols_to_drop + add_drop
        self.X_val.drop(columns=all_drop, inplace=True)

        print(f"  Dropped columns {all_drop} from X_val.")

    def add_rank(self, book_score=5, click_score=1):
        """ 
        Adds a score attribute to the dataframe
        using the click and booking attributes.
        """

        def rank(x):
            if x["booking_bool"] == 1:
                return book_score
            elif x["click_bool"] == 1:
                return click_score
            return 0

        self.train["rank"] = self.train.apply(lambda x: rank(x), axis=1)
        self.val["rank"] = self.val.apply(lambda x: rank(x), axis=1)

    def add_qid(self, att, sort=True):
        """
        Renames the query column to qid and
        optionally sorts df by this col.
        """
        self.train.rename(columns={att: "qid"}, inplace=True)
        self.test.rename(columns={att: "qid"}, inplace=True)
        self.val.rename(columns={att: "qid"}, inplace=True)

        if sort:
            self.train.sort_values(by="qid", inplace=True)
            self.test.sort_values(by="qid", inplace=True)
            self.val.sort_values(by="qid", inplace=True)

        # unnecessary

    def prep_data(self, drop_cols=False):
        """
        Split loaded data into features, labels and groups.
        """
        self.X_trn = self.train.loc[:, ~self.train.columns.isin(["qid", "rank"])].copy()
        self.y_trn = self.train.loc[:, "rank"].copy()
        self.qid_trn = self.train.loc[:, "qid"].copy()
        self.g_trn = self.groups(self.qid_trn)

        self.X_tst = self.test.loc[:, ~self.test.columns.isin(["qid", "rank"])].copy()
        self.qid_tst = self.train.loc[:, "qid"].copy()

        self.X_val = self.val.loc[:, ~self.val.columns.isin(["qid", "rank"])].copy()
        self.y_val = self.val.loc[:, "rank"].copy()
        self.qid_val = self.val.loc[:, "qid"].copy()
        self.g_val = self.groups(self.qid_val)

        if drop_cols:
            self.drop_redundant_cols()

        self.train_DMatrix = xgb.DMatrix(self.X_trn, self.y_trn)
        self.train_DMatrix.set_group(self.g_trn)

        self.val_DMatrix = xgb.DMatrix(self.X_val, self.y_val)
        self.val_DMatrix.set_group(self.g_val)

        self.tst_DMatrix = xgb.DMatrix(self.X_tst)

    def fit_XGB(self, tr_idx, val_idx, params, grid, split, verbose=False):
        """
        Fits XGBRanker on a subset of indices.
        Returns the NDCG score.
        """
        params_to_save = params.copy()
        start = time.time()

        rounds = params.pop("num_boost_round")
        esr = params.pop("early_stopping_rounds")

        model = xgb.train(
            params,
            self.train_DMatrix,
            num_boost_round=rounds,
            evals=[(self.val_DMatrix, "validation")],
            early_stopping_rounds=esr,
            verbose_eval=verbose,
        )

        score = model.attributes()["best_score"]

        # score = custom_ndcg_score_groups(self.y_val, y_val_pred, self.g_val, k=5)
        pparams = {
            key: round(params_to_save[key], 2)
            for key in [
                "max_depth",
                "eta",
                "gamma",
                "alpha",
                "reg_lambda",
                "num_parallel_tree",
                "subsample",
                "colsample_bytree",
                "min_child_weight",
                # "num_boost_round",
            ]
        }

        end = time.time()

        print(f"{grid} - {round(end-start,1)}s NDCG@5: {score} - {pparams}")
        params_to_save["score"] = score
        params_to_save["id"] = grid

        return params_to_save

    def gridsearch(
        self,
        params_options,
        out_path=False,
        multip=False,
        n_rounds=None,
        verbose=False,
        cores=-1,
    ):
        """
        Runs gridsearch with supplied options. 
        """
        if multip and cores == -1:
            cores = mp.cpu_count()
            print(f"{cores} cores found. Using all")

        self.get_permutations(params_options, n_rounds=n_rounds)
        grids = len(self.grid_permutations)

        print(f"Starting {grids} runs.")

        results = []
        args = []

        for gid, params in enumerate(self.grid_permutations):
            if multip:
                args.append([0, 0, params, gid, 1])
            else:
                results.append(self.fit_XGB(0, 0, params, gid, 1, verbose))

        if multip:
            with mp.Pool(cores) as pool:
                results = pool.starmap(self.fit_XGB, args)

        self.results = pd.DataFrame(results)

        if out_path:
            self.results.to_pickle(out_path)

    def get_best_params(self):
        """
        Reads results dataframe and sorts by score.
        Sets self.best_params to the best parameters.
        """
        # # assert self.results, "Run gridsearch or load results first."
        # groupby_columns = list(self.results.columns)

        # for col in ["split_id", "score"]:
        #     groupby_columns.remove(col)

        # grouped = self.results.groupby(groupby_columns).mean().reset_index()
        # grouped.drop(columns=["split_id"], inplace=True)

        self.results.sort_values(by=["score"], inplace=True)

        self.best_params = self.results.iloc[-1].to_dict()
        # self.best_params = best
        return self.best_params

    def train_best_model(self, outfile, custom_hyper=None, save_model=None):
        """
        Trains model on the full training set
        using either custom hyperparams- or
        best available through gridsearch.
        """
        if custom_hyper is not None:
            params = custom_hyper.copy()
        else:
            params = self.best_params.copy()

        rounds = params.pop("num_boost_round")
        esr = params.pop("early_stopping_rounds")

        model = xgb.train(
            params,
            self.train_DMatrix,
            num_boost_round=rounds,
            evals=[(self.val_DMatrix, "validation")],
            early_stopping_rounds=esr,
            verbose_eval=True,
        )

        self.best_model = model

        output = self.test.copy()
        output["scores"] = model.predict(self.tst_DMatrix)

        # Write to sorted csv
        output = output.sort_values(by=["qid", "scores"], ascending=[True, False])
        output.rename(columns={"qid": "srch_id"}, inplace=True)

        kaggle_df = output[["srch_id", "prop_id"]].copy()
        kaggle_df.to_csv(outfile, index=False)

    def save_results(self, y_true, y_preds, qid):
        self.results_df = pd.DataFrame(
            {"predictions": y_preds, "truth": y_true, "groups": qid}
        )

    def save_model(self, fpath):
        self.best_model.save_model(fpath)

    def save_model_meta(self, fpath):
        out_obj = dict(
            fi=self.best_model.feature_importances_,
            names=self.X_trn.columns,
            model=self.best_model,
            X_val=self.X_val,
            y_val=self.y_val,
        )

        pickle.dump(out_obj, open(fpath, "wb"))

    @staticmethod
    def groups(input):
        """
        Counts groupsize. 
        Expects sorted qids.
        Example:
        Input: [1,1,5,5,5,11,11,29]
        Output: [2,3,2,1]
        """
        if not type(input) == list:
            input = list(input)

        groups = [0]
        current = input[0]

        for qid in input:
            if qid == current:
                groups[-1] += 1
            else:
                current = qid
                groups.append(1)

        return groups

    def get_permutations(self, param_dict, n_rounds=False):
        """
        Splits a dict with key: lists into a
        list of dicts containing all unique
        key:value permutations.
        """
        keys, values = zip(*param_dict.items())
        self.grid_permutations = [dict(zip(keys, v)) for v in it.product(*values)]

        if n_rounds:
            # If n_rounds > len(grid_perm)
            # then select all options.
            permcount = len(self.grid_permutations)
            if n_rounds > permcount:
                n_rounds = permcount
                print(f"selected all {n_rounds} permutations")
            self.grid_permutations = random.sample(self.grid_permutations, n_rounds)
