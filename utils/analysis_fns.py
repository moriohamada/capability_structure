import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from config import ANALYSIS_OPS
from tqdm.notebook import tqdm
from typing import List
import statsmodels.api as sm
from sklearn.model_selection import LeaveOneOut

def empirical_ci(data: np.ndarray = None, ci_size: float = 99) -> (float, float):
    prctile = (100-ci_size)/2
    ci_l = np.percentile(data, prctile, axis=0)
    ci_u = np.percentile(data, 100 - prctile, axis=0)
    return ci_l, ci_u

def knn_impute(X: np.ndarray = None, n_neighbours: int = ANALYSIS_OPS['kNN_k']) -> np.ndarray:

    X_imputed = KNNImputer(n_neighbors=n_neighbours, weights='distance', missing_values=np.nan).fit_transform(X)

    return X_imputed

def pca_scores(X: np.ndarray, n_components: int = None):
    """
    Fit PCA and return scores, components, and explained variance.
    """
    pca = PCA(n_components=n_components).fit(X)
    scores = pca.transform(X)
    return scores, pca.components_, pca.explained_variance_

def top_loadings(weights: np.ndarray,
                 var_names: list[str],
                 n_top: int = 8,
                 start_idx: int = 0):
    n_vars, n_comp = weights.shape
    rows = []
    for c in range(n_comp):
        print(f"\nPC{c + start_idx + 1}")
        ranked = np.argsort(np.abs(weights[:, c]))[::-1][:n_top]
        for idx in ranked:
            print(f'  {var_names[idx]:>40s}  {weights[idx, c]:+.3f}')
            rows.append({'component': c + start_idx + 1, 'variable': var_names[idx], 'loading': weights[idx, c]})
    return pd.DataFrame(rows)

def parallel_analysis(X: np.ndarray,
                      n_shuffles: int = 1000,
                      seed: int = 0):
    """
    Parallel analysis for PCA.

    Compares real eigenvalues against those from benchmark-shuffled data.
    Parameters
    ----------
    X : (n_models, n_bench)

    Returns
    -------
    real_var : (n_vars,) explained variance per component
    null_vars : (n_shuffles, n_vars) explained variance per component per shuffle
    """
    rng = np.random.default_rng(seed)
    n_samples, n_vars = X.shape

    real_var = PCA().fit(X).explained_variance_

    null_vars = np.zeros((n_shuffles, n_vars))
    for s in range(n_shuffles):
        X_shuf = X.copy()
        for j in range(n_vars):
            rng.shuffle(X_shuf[:, j])
        null_vars[s] = PCA().fit(X_shuf).explained_variance_

    return real_var, null_vars



def pca_cv(X: np.ndarray,
           n_splits: int = 100,
           n_shuffles: int = 100,
           holdout_frac: float = 0.1,
           seed: int = 0):
    """
    Cross-validated variance explained per PCA component, with
    column-shuffled null distribution.

    Holds out random rows, fits PCA on the rest, projects held-out rows
    onto each component. Null: shuffle held-out data columns to destroy
    correlations, then project onto the real components.

    Parameters
    ----------
    X : (n_samples, n_vars)
    n_splits : number of random train/test splits
    n_shuffles : number of column shuffles per split
    holdout_frac : fraction of rows to hold out
    seed : random seed

    Returns
    -------
    real_var : (n_splits, n_vars) held-out variance per component
    null_var : (n_splits, n_vars, n_shuffles) null variance per component
    """
    rng = np.random.default_rng(seed)
    n_samples, n_vars = X.shape
    n_holdout = max(1, int(n_samples * holdout_frac))

    real_var = np.zeros((n_splits, n_vars))
    null_var = np.zeros((n_splits, n_vars, n_shuffles))

    for s in range(n_splits):
        idx = rng.permutation(n_samples)
        test_idx, train_idx = idx[:n_holdout], idx[n_holdout:]

        pca = PCA(n_components=n_vars).fit(X[train_idx])
        X_test_centered = X[test_idx] - pca.mean_

        for k in range(n_vars):
            scores = X_test_centered @ pca.components_[k]
            real_var[s, k] = np.var(scores)

        for sh in range(n_shuffles):
            X_shuf = X_test_centered.copy()
            for j in range(n_vars):
                rng.shuffle(X_shuf[:, j])
            for k in range(n_vars):
                null_scores = np.dot(X_shuf, pca.components_[k])
                null_var[s, k, sh] = np.var(null_scores)

    return real_var, null_var


def permutation_test_deltaR2(y, X_base, X_extra, n_perms=1000, seed=0):
    """
    Permutation test for whether additional regressors (X_extra) improves R2 beyond base model (X_base).

    Shuffles rows of X_extra to break association with y, refits, and builds a null distribution of delta R2.

    Returns: observed ΔR², null distribution, p-value, full model OLS result
    """
    rng = np.random.default_rng(seed)

    X_full = sm.add_constant(np.hstack([X_base, X_extra]))
    X_restricted = sm.add_constant(X_base)

    m_restricted = sm.OLS(y, X_restricted).fit()
    m_full = sm.OLS(y, X_full).fit()
    observed = m_full.rsquared - m_restricted.rsquared

    null = np.zeros(n_perms)
    for i in range(n_perms):
        X_extra_shuf = X_extra.copy()
        rng.shuffle(X_extra_shuf)
        X_perm = sm.add_constant(np.hstack([X_base, X_extra_shuf]))
        null[i] = sm.OLS(y, X_perm).fit().rsquared - m_restricted.rsquared

    p = np.mean(null >= observed)
    return observed, null, p, m_full

def permutation_test_loo(y, X_base, X_extra, n_perms=100, seed=0):
    """
    Permutation test on LOO ΔR²: does adding X_extra improve
    out-of-sample prediction beyond X_base?

    Returns: observed LOO ΔR², null distribution, p-value, full model OLS result
    """
    rng = np.random.default_rng(seed)
    loo = LeaveOneOut()

    X_full = sm.add_constant(np.hstack([X_base, X_extra]))

    def loo_r2(X):
        preds = np.zeros(len(y))
        for train_idx, test_idx in loo.split(y):
            preds[test_idx] = sm.OLS(y[train_idx], X[train_idx]).fit().predict(X[test_idx])
        return 1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2)

    r2_base = loo_r2(sm.add_constant(X_base))
    r2_full = loo_r2(X_full)
    observed = r2_full - r2_base

    null = np.zeros(n_perms)
    for i in range(n_perms):
        X_extra_shuf = X_extra.copy()
        rng.shuffle(X_extra_shuf)
        X_perm = sm.add_constant(np.hstack([X_base, X_extra_shuf]))
        null[i] = loo_r2(X_perm) - r2_base

    p = np.mean(null >= observed)
    m_full = sm.OLS(y, X_full).fit()
    return observed, null, p, m_full