#%%
import itertools as it
import multiprocessing as mp

import pickle
import random

import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score

from lib_ndcg import custom_ndcg_score_groups
from lib_data import *


class LearnToRank:
    custom_validation = False

    def load_data(self, train_df_pickle, test_df_pickle, val_df_pickle=False):
        """
        Loads data from picklefiles.
        """
        self.train = pd.read_pickle(train_df_pickle)
        self.test = pd.read_pickle(test_df_pickle)

        print("Loaded data..")
        print("  Train shape:", self.train.shape)
        print("  Test shape:", self.test.shape)

        if val_df_pickle != False:
            self.val = pd.read_pickle(val_df_pickle)
            self.custom_validation = True
            self.splits = [1]
            print("  Val shape:", self.val.shape)

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

        cols_to_drop = [x for x in train_columns if x not in test_columns]
        self.X_trn.drop(columns=cols_to_drop, inplace=True)
        print(f"  Dropped columns {cols_to_drop} from X_train.")

        if self.custom_validation:
            val_columns = self.X_val.columns
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

        if self.custom_validation:
            self.val["rank"] = self.val.apply(lambda x: rank(x), axis=1)

    def add_qid(self, att, sort=True):
        """
        Renames the query column to qid and
        optionally sorts df by this col.
        """
        self.train.rename(columns={att: "qid"}, inplace=True)
        self.test.rename(columns={att: "qid"}, inplace=True)

        if self.custom_validation:
            self.val.rename(columns={att: "qid"}, inplace=True)

        if sort:
            self.train.sort_values(by="qid", inplace=True)
            self.test.sort_values(by="qid", inplace=True)

            if self.custom_validation:
                self.val.sort_values(by="qid", inplace=True)

        # unnecessary

    def prep_data(self, drop_cols=False):
        """
        Split loaded data into features, labels and groups.
        """
        self.X_trn = self.train.loc[:, ~self.train.columns.isin(["qid", "rank"])].copy()
        self.y_trn = self.train.loc[:, "rank"].copy()
        self.qid_trn = self.train.loc[:, "qid"].copy()

        self.X_tst = self.test.loc[:, ~self.test.columns.isin(["qid", "rank"])].copy()
        self.qid_tst = self.train.loc[:, "qid"].copy()

        if self.custom_validation:
            self.X_val = self.val.loc[:, ~self.val.columns.isin(["qid", "rank"])].copy()
            self.y_val = self.val.loc[:, "rank"].copy()
            self.qid_val = self.val.loc[:, "qid"].copy()

        if drop_cols:
            self.drop_redundant_cols()

        assert len(self.X_trn.columns) == len(
            self.X_tst.columns
        ), "Columns X train & X test not identical. Set drop_redundant to True to solve."

    def create_groupsplits(self, test_size=0.2, splits=1):
        """
        Splits the datasets n times, enforcing
        group seperation. 
        """
        if self.custom_validation:
            print("Note: validation set supplied. Skipping groupsplits.")
        else:
            gss = GroupShuffleSplit(test_size=test_size, n_splits=splits)
            self.splits = list(gss.split(self.X_trn, groups=self.qid_trn))

    def fit_XGB(self, tr_idx, val_idx, params, grid, split):
        """
        Fits XGBRanker on a subset of indices.
        Returns the NDCG score.
        """

        if self.custom_validation:
            X_train = self.X_trn
            y_train = self.y_trn
            g_train = self.groups(self.qid_trn)

            X_val = self.X_val
            y_val = self.y_val
            qid_val = self.qid_val
            g_val = self.groups(qid_val)
        else:
            X_train = self.X_trn.iloc[tr_idx]
            y_train = self.y_trn.iloc[tr_idx]
            g_train = self.groups(self.qid_trn.iloc[tr_idx])

            X_val = self.X_trn.iloc[val_idx]
            y_val = self.y_trn.iloc[val_idx]
            qid_val = self.qid_trn.iloc[val_idx]
            g_val = self.groups(qid_val)

        model = xgb.XGBRanker(**params)
        model.fit(X_train, y_train, group=g_train, verbose=True)
        y_val_pred = model.predict(X_val)

        # self.save_results(y_val, y_pred, qid_val)
        # score = ndcg_score([y_val, qid_val], [y_val_pred, qid_val], k=5)
        # print("NDCG scikitlearn: ", score)

        score = custom_ndcg_score_groups(y_val, y_val_pred, g_val, k=5)
        pparams = {
            key: params[key]
            for key in [
                "objective",
                "eta",
                "max_depth",
                "n_estimators",
                "subsample",
                "colsample_bytree",
            ]
        }
        print(f"G: {grid} / S: {split} NDCG@5: {round(score,4)}  {pparams}")

        rval = dict(params=str(params), grid_id=grid, split_id=split, score=score,)
        return rval

    def gridsearch(self, params_options, out_path=False, draw_n_random=False):
        """
        Runs gridsearch with supplied options. 
        """
        self.get_permutations(params_options, draw_n_random)
        grids = len(self.grid_permutations)
        splits = len(self.splits)

        print(f"{grids} param settings  x  {splits} splits  = {grids*splits} runs")

        results = []

        for gid, params in enumerate(self.grid_permutations):
            if self.custom_validation:
                results.append(self.fit_XGB(0, 0, params, gid, 1))
            else:
                for sid, (tr_idx, val_idx) in enumerate(self.splits):
                    results.append(self.fit_XGB(tr_idx, val_idx, params, gid, sid + 1))

        self.results = pd.DataFrame(results)

        if out_path:
            self.results.to_pickle(out_path)

    def get_best_params(self):
        """
        Reads results dataframe and sorts by score.
        Sets self.best_params to the best parameters.
        """
        # assert self.results, "Run gridsearch or load results first."
        groupby_columns = list(self.results.columns)

        for col in ["split_id", "score"]:
            groupby_columns.remove(col)

        grouped = self.results.groupby(groupby_columns).mean().reset_index()
        grouped.drop(columns=["split_id"], inplace=True)
        grouped.sort_values(by=["score"], inplace=True)

        self.gridsearch_results = grouped

        best = grouped.iloc[-1]
        self.best_params = eval(best["params"])

        return grouped

    def train_best_model(self, outfile, custom_hyper=None, save_model=None):
        """
        Trains model on the full training set
        using either custom hyperparams- or
        best available through gridsearch.
        """
        if custom_hyper is not None:
            params = custom_hyper
        else:
            params = self.best_params

        model = xgb.XGBRanker(**params)

        X_train = self.X_trn
        y_train = self.y_trn
        g_train = self.groups(self.qid_trn)

        model.fit(X_train, y_train, group=g_train, verbose=True)

        self.best_model = model
        # if save_model is not None:

        # Predict on the kaggle set
        output = self.test.copy()
        output["scores"] = model.predict(self.X_tst)

        # Write to sorted csv
        output = output.sort_values(by=["qid", "scores"], ascending=[True, False])
        output.rename(columns={"qid": "srch_id"}, inplace=True)

        kaggle_df = output[["srch_id", "prop_id"]].copy()
        kaggle_df.to_csv(outfile, index=False)

    def save_results(self, y_true, y_preds, qid):
        self.results_df = pd.DataFrame(
            {"predictions": y_preds, "truth": y_true, "groups": qid}
        )

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

    def get_permutations(self, param_dict, draw_n_random=False):
        """
        Splits a dict with key: lists into a
        list of dicts containing all unique
        key:value permutations.
        """
        keys, values = zip(*param_dict.items())
        self.grid_permutations = [dict(zip(keys, v)) for v in it.product(*values)]

        if isinstance(draw_n_random, int):
            self.grid_permutations = random.sample(self.grid_permutations, draw_n_random)
