#%%
import itertools as it
import multiprocessing as mp

import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score

from lib_data import *


class LearnToRank:
    def load_data(self, train_df_pickle, test_df_pickle):
        self.train = pd.read_pickle(train_df_pickle)
        self.test = pd.read_pickle(test_df_pickle)
        print("Loaded data..")
        print("Train shape:", self.train.shape)
        print("Test shape:", self.test.shape)

    def load_results(self, results_df_pickle):
        print("Loaded results")
        self.results = pd.read_pickle(results_df_pickle)

    def drop_redundant_cols(self):
        """
        Drops columns from training data
        if not in testing data.
        """
        test_columns = self.X_tst.columns
        train_columns = self.X_trn.columns

        cols_to_drop = [x for x in train_columns if x not in test_columns]
        self.X_trn.drop(columns=cols_to_drop, inplace=True)

        print(f"Dropped columns {cols_to_drop} from X_train.")

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

    def add_qid(self, att, sort=True):
        """
        Renames the query column to qid and
        optionally sorts df by this col.
        """
        self.train.rename(columns={att: "qid"}, inplace=True)
        self.test.rename(columns={att: "qid"}, inplace=True)

        if sort:
            self.train.sort_values(by="qid", inplace=True)
            self.test.sort_values(by="qid", inplace=True)

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

        if drop_cols:
            self.drop_redundant_cols()

        assert len(self.X_trn.columns) == len(
            self.X_tst.columns
        ), "Columns X train & X test not identical. Set drop_redundant to True to solve."

    def create_groupsplits(self, test_size=0.2, splits=1, seed=42):
        gss = GroupShuffleSplit(test_size=test_size, n_splits=splits, random_state=seed)
        self.splits = list(gss.split(self.X_trn, groups=self.qid_trn))

    def fit_XGB(self, tr_idx, val_idx, fit_params, grid_params, grid_id, split_id):
        """
        Fits XGBRanker on a subset of indices.
        Returns the NDCG score.
        """

        model = xgb.XGBRanker(**fit_params, **grid_params)

        X_train = self.X_trn.iloc[tr_idx]
        y_train = self.y_trn.iloc[tr_idx]
        g_train = self.groups(self.qid_trn.iloc[tr_idx])

        X_val = self.X_trn.iloc[val_idx]
        y_val = self.y_trn.iloc[val_idx]
        qid_val = self.qid_trn.iloc[val_idx]

        model.fit(X_train, y_train, group=g_train, verbose=True)
        y_pred = model.predict(X_val)

        self.save_results(y_val, y_pred, qid_val)

        score = ndcg_score([y_val, qid_val], [y_pred, qid_val], k=5)

        print(
            f"GridID: {grid_id} / Split: {split_id} [SCORE]: {round(score,5)}  {grid_params}"
        )

        rval = dict(
            grid_params=str(grid_params),
            fit_params=str(fit_params),
            grid_id=grid_id,
            split_id=split_id,
            score=score,
        )
        return rval

    def gridsearch_mp(self, fit_params, grid_params_options, out_path=False, cores=4):
        """
        Runs gridsearch. 
        """
        self.get_permutations(grid_params_options)
        grids = len(self.grid_permutations)
        splits = len(self.splits)

        print(f"{grids} param settings  x  {splits} splits  = {grids*splits} runs")

        input = []

        for gid, grid_params in enumerate(self.grid_permutations):
            for sid, (tr_idx, val_idx) in enumerate(self.splits):
                input.append((tr_idx, val_idx, fit_params, grid_params, gid, sid + 1))

        with mp.Pool(processes=cores) as pool:
            results = pool.starmap(self.fit_XGB, input)

        self.results = pd.DataFrame(results)

        if out_path:
            self.results.to_pickle(out_path)

    def gridsearch(self, fit_params, grid_params_options, out_path=False):
        """
        Runs gridsearch. 
        """
        self.get_permutations(grid_params_options)
        grids = len(self.grid_permutations)
        splits = len(self.splits)

        print(f"{grids} param settings  x  {splits} splits  = {grids*splits} runs")

        results = []

        for gid, grid_params in enumerate(self.grid_permutations):
            for sid, (tr_idx, val_idx) in enumerate(self.splits):
                results.append(
                    self.fit_XGB(tr_idx, val_idx, fit_params, grid_params, gid, sid + 1)
                )

        self.results = pd.DataFrame(results)

        if out_path:
            self.results.to_pickle(out_path)

    def get_best_params(self):
        # assert self.results, "Run gridsearch first"
        groupby_columns = list(ranker.results.columns)

        for col in ["split_id", "score"]:
            groupby_columns.remove(col)

        grouped = self.results.groupby(groupby_columns).mean().reset_index()
        grouped.drop(columns=["split_id"], inplace=True)
        grouped.sort_values(by=["score"], inplace=True)

        self.results_aggregated = grouped

        best = grouped.iloc[-1]
        self.best_grid_params = eval(best["grid_params"])
        self.best_fit_params = eval(best["fit_params"])

        return grouped

    def train_best_model(self, outfile):
        model = xgb.XGBRanker(**self.best_fit_params, **self.best_grid_params)
        X_train = self.X_trn
        y_train = self.y_trn
        g_train = self.groups(self.qid_trn)

        model.fit(X_train, y_train, group=g_train, verbose=True)

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

    def get_permutations(self, param_dict):
        """
        Splits a dict with key: lists into a
        list of dicts containing all unique
        key:value permutations.
        """
        keys, values = zip(*param_dict.items())
        self.grid_permutations = [dict(zip(keys, v)) for v in it.product(*values)]
