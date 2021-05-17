# %%
import numpy as np
import multiprocessing as mp
import pickle
from sklearn.metrics import ndcg_score

from lib_LTR import *


def custom_dcg_score(y_true, y_score, k=20, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true: array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score: array-like, shape = [n_samples]
        Predicted scores.
    k: int
        Rank.
    gains: str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k: float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def custom_ndcg_score(y_true, y_score, k=20, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true: array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score: array-like, shape = [n_samples]
        Predicted scores.
    k: int
        Rank.
    gains: str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k: float
    """
    best = custom_dcg_score(y_true, y_true, k, gains)
    actual = custom_dcg_score(y_true, y_score, k, gains)
    return actual / best


def custom_ndcg_score_groups(
    y_true, y_score, groups, k=20, gains="exponential", list=False
):
    """
    Extract 
    """
    start_idx = 0
    pool = mp.Pool(10)
    args = []

    for g in groups:
        end_idx = start_idx + g
        true = y_true[start_idx:end_idx]
        score = y_score[start_idx:end_idx]
        start_idx = end_idx
        args.append([true, score, k, gains])

    results = pool.starmap(custom_ndcg_score, args)

    if list:
        return results

    return np.mean(results)


#%%
def main():
    # y_true = [0, 5, 1, 0, 1, 0, 0]
    # y_lab = [0, 0, 0, 0, 1, 1, 1]
    # groups = LearnToRank.groups(y_lab)
    # y_pred = [0, 5, 1, 0, 1, 0, 0]

    d = pickle.load(open("output/out.p", "rb"))
    lbl = d["lbl"]
    groups = LearnToRank.groups(lbl)
    t = d["true"]
    s = d["score"]

    x = custom_ndcg_score_groups(t, s, groups, k=5, list=False)


if __name__ == "__main__":
    main()
