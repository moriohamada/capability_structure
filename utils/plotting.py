import os
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
sns.set_style("whitegrid")
import numpy as np
import pandas as pd
from utils.analysis_fns import empirical_ci
import statsmodels.api as sm
from sklearn.model_selection import LeaveOneOut
from typing import List

def save_fig(ax: Axes, path: str = None, type: str = 'svg'):
    if path is None:
        dir = './figures/'
        os.makedirs(dir, exist_ok=True)
        path = dir + 'plot'
    fig = ax.get_figure()
    if type == 'svg':
        fig.savefig(f"{path}.svg", bbox_inches='tight')
    elif type == 'png':
        fig.savefig(f"{path}.png", bbox_inches='tight', dpi=500)

def visualize_parallel_analysis_results(real_var: np.ndarray,
                                        null_var: np.ndarray,
                                        ci_size: float = 99,
                                        convert_to_expvar: bool = False,
                                        xlabel: str = 'Component'):

    fig, ax = plt.subplots(figsize=(4, 4))
    n = len(real_var)
    components = np.arange(1, n + 1)

    if convert_to_expvar:
        real_var = real_var/np.sum(real_var) * 100
        null_var = null_var/(np.sum(null_var, axis=1, keepdims=True) )* 100

    ci_l, ci_u = empirical_ci(null_var, ci_size=ci_size)
    med = np.percentile(null_var, 50, axis=0)

    ax.fill_between(components, ci_l, ci_u, alpha=0.25, color='grey', label=f'{ci_size}% CI (null)')
    sns.lineplot(x=components, y=med, marker='s', linestyle='--', color='grey', label='Null median', ax=ax)
    sns.lineplot(x=components, y=real_var, marker='o', label='Real data', ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Eigenvalue' if not convert_to_expvar else '% variance explained')
    ax.set_xticks(components)
    ax.legend()
    sns.despine()

    n_factors = np.sum(real_var > ci_u)
    print(f"Number of {xlabel.lower()}s above {ci_size}% CI: {n_factors}")

def plot_cv_parallel(real_var: np.ndarray,
                     null_var: np.ndarray,
                     ci_size: float = 99):
    """
    Parameters
    ----------
    real_var : (n_splits, n_vars)
    null_var : (n_splits, n_vars, n_shuffles)
    """
    fig, ax = plt.subplots(figsize=(4, 4))

    real_mean = real_var.mean(axis=0)
    null_flat = null_var.reshape(-1, null_var.shape[1])  # pool splits and shuffles
    ci_l, ci_u = empirical_ci(null_flat, ci_size=ci_size)
    null_med = np.median(null_flat, axis=0)

    components = np.arange(1, len(real_mean) + 1)
    ax.fill_between(components, ci_l, ci_u, alpha=0.25, color='grey', label=f'{ci_size}% CI (null)')
    ax.plot(components, null_med, marker='s', linestyle='--', color='grey', label='Null median')
    ax.plot(components, real_mean, marker='o', label='Real (held-out)')
    ax.set_xlabel('Component')
    ax.set_ylabel('Held-out variance explained')
    ax.set_xticks(components)
    ax.legend()
    sns.despine()

    n_sig = np.sum(real_mean > ci_u)
    print(f"Components above {ci_size}% CI: {n_sig}")
    return ax

def loadings_heatmap(weights: np.ndarray, var_names: list[str], comp_labels: list[str] = None):
    n_vars, n_comp = weights.shape
    if comp_labels is None:
        comp_labels = [f"PC{i+1}" for i in range(n_comp)]
    order = np.lexsort((-np.max(np.abs(weights), axis=1), np.argmax(np.abs(weights), axis=1)))
    df = pd.DataFrame(weights[order], index=np.array(var_names)[order],
                       columns=comp_labels)

    fig, ax = plt.subplots(figsize=(max(4, n_comp), max(5, n_vars * 0.3)))
    vmax = np.abs(weights).max()
    sns.heatmap(df, center=0, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
                annot=True, fmt='.2f', linewidths=0.5, ax=ax)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    return ax

def plot_scores_1d(values, df_info, has_mask, xlabel='Score', n_label=15):
    fig, ax = plt.subplots(figsize=(10, 2))

    rng = np.random.default_rng()
    jitter = rng.uniform(-0.2, 0.2, len(values))

    creator = df_info.loc[has_mask, 'creator'].reset_index(drop=True)
    top5 = creator.value_counts().head(5).index
    creator = creator.where(creator.isin(top5), other='Other')

    for name, idx in creator.groupby(creator).groups.items():
        ax.scatter(values[idx], jitter[idx], label=name, alpha=0.6, s=20)

    order = np.argsort(values)
    label_idx = np.concatenate([order[:n_label // 2], order[-(n_label - n_label // 2):]])
    for i in label_idx:
        ax.annotate(df_info.loc[has_mask, 'slug'].iloc[i], (values[i], jitter[i]),
                    fontsize=6, alpha=0.8, xytext=(0, 6), textcoords='offset points', ha='center')

    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.axvline(np.mean(values), color='grey', linewidth=0.5, linestyle='--')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    return ax

def plot_component_1d(scores, comp_idx, df_info, comp_name=None, n_label=15):
    fig, ax = plt.subplots(figsize=(10, 2))

    if comp_name is None:
        comp_name = f"PC{comp_idx + 1}"

    y = scores[:, comp_idx] if scores.ndim > 1 else scores

    creator = df_info['creator'].copy()
    top_creators = creator.value_counts().head(5).index
    creator = creator.where(creator.isin(top_creators), other='Other')

    rng = np.random.default_rng()
    jitter = rng.uniform(-0.2, 0.2, len(y))

    for name, idx in creator.groupby(creator).groups.items():
        ax.scatter(y[idx], jitter[idx], label=name, alpha=0.6, s=20)

    order = np.argsort(y)
    label_idx = np.concatenate([order[:n_label // 2], order[-(n_label - n_label // 2):]])
    for i in label_idx:
        ax.annotate(df_info['slug'].iloc[i], (y[i], jitter[i]),
                    fontsize=6, alpha=0.8, xytext=(0, 6), textcoords='offset points',
                    ha='center')

    ax.set_xlabel(comp_name)
    ax.set_yticks([])
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    return ax


def plot_component_v_time(scores, comp_idx,
                          comp_name: str = None,
                          df_info: pd.DataFrame = None,
                          n_label: int = 5):
    fig, ax = plt.subplots(figsize=(6, 3))

    if comp_name is None:
        comp_name = f"PC{comp_idx + 1}"

    dates = pd.to_datetime(df_info['release_date'], errors='coerce')
    y = scores[:, comp_idx] if scores.ndim > 1 else scores

    creator = df_info['creator'].copy()
    top_creators = creator.value_counts().head(5).index
    creator = creator.where(creator.isin(top_creators), other='Other')

    for name, group in pd.DataFrame({'date': dates, 'score': y, 'creator': creator}).groupby('creator'):
        mask = group['date'].notna()
        ax.scatter(group.loc[mask, 'date'], group.loc[mask, 'score'], label=name, alpha=0.6, s=30)

        valid = dates.notna()
        if n_label > 0 and valid.any():
            extreme_idx = np.argsort(y[valid])
            label_idx = np.concatenate([extreme_idx[:n_label // 2], extreme_idx[-(n_label // 2):]])
            valid_positions = np.where(valid)[0]
            for i in label_idx:
                pos = valid_positions[i]
                ax.annotate(df_info['slug'].iloc[pos], (dates.iloc[pos], y[pos]),
                            fontsize=7, alpha=0.4, xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('Release date')
    ax.set_ylabel(comp_name)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    return ax


def plot_component_scatter(scores,
                           cx: int = 0, cy: int = 1,
                           df_info: pd.DataFrame = None,
                           comp_names: List[str] = None,
                           label_models: int | List[str] = None):
    fig, ax = plt.subplots(figsize=(4, 4))

    if comp_names is None:
        comp_names = [f"PC{i + 1}" for i in range(scores.shape[1])]

    x = scores[:, cx]
    y = scores[:, cy]

    creator = df_info['creator'].copy()
    top_creators = creator.value_counts().head(5).index
    creator = creator.where(creator.isin(top_creators), other='Other')

    for name, group_idx in pd.Series(creator).groupby(creator).groups.items():
        ax.scatter(x[group_idx], y[group_idx], label=name, alpha=0.6, s=30)

    ax.set_xlabel(comp_names[cx])
    ax.set_ylabel(comp_names[cy])
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')

    if isinstance(label_models, int):
        dist = x ** 2 + y ** 2
        top_idx = np.argsort(dist)[-label_models:]
        label_models = df_info['slug'].iloc[top_idx].tolist()

    if label_models is not None:
        for i, name in enumerate(df_info['slug']):
            if name in label_models:
                ax.annotate(name, (x[i], y[i]), fontsize=6, alpha=0.8,
                            xytext=(4, 4), textcoords='offset points')

    return ax

def plot_pc_scatter_honesty(scores, cx, cy, y, df_info, has_mask,
                            creator_filter=None, comp_names=None, n_label=10):
    fig, ax = plt.subplots(figsize=(6, 5))

    mask = has_mask.copy()
    if creator_filter:
        mask = mask & (df_info['creator'] == creator_filter).values

    x_vals = scores[mask, cx]
    y_vals = scores[mask, cy]
    h_vals = y[df_info.loc[has_mask, 'creator'].values == creator_filter] if creator_filter else y
    slugs = df_info.loc[mask, 'slug'].values

    sc = ax.scatter(x_vals, y_vals, c=h_vals, cmap='vanimo', s=40, edgecolors='k', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label='Honesty score')

    if n_label > 0:
        per_axis = n_label // 2
        order_x = np.argsort(x_vals)
        order_y = np.argsort(y_vals)
        label_idx = set(np.concatenate([
            order_x[:per_axis], order_x[-per_axis:],
            order_y[:per_axis], order_y[-per_axis:],
        ]))
        for i in label_idx:
            ax.annotate(slugs[i], (x_vals[i], y_vals[i]),
                        fontsize=6, alpha=0.7, xytext=(4, 4), textcoords='offset points')

    if comp_names is None:
        comp_names = [f"PC{i+1}" for i in range(scores.shape[1])]

    ax.set_xlabel(comp_names[cx])
    ax.set_ylabel(comp_names[cy])
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
    if creator_filter:
        ax.set_title(creator_filter)
    fig.tight_layout()
    return ax