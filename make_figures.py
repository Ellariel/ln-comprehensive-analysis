import sys
import pickle
import inspect
import os, argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
from numbers import Number
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from scripts.utils import gini_coefficient, set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", default=1, type=int)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--results_dir", default=None, type=str)
    parser.add_argument("--shapes_file", default="shapes_fix.csv", type=str)
    args = parser.parse_args()
else:
    sys.exit()

base_dir = os.path.dirname(__file__)
if "app" in base_dir:
    base_dir = "./"
data_dir = os.path.join(base_dir, "data") if args.data_dir is None else args.data_dir
results_dir = (
    os.path.join(base_dir, "results") if args.results_dir is None else args.results_dir
)
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("data_dir:", data_dir)
print("results_dir:", results_dir)


def resample_with_ci(series, rule="2W", z=1.96):
    r = series.asfreq("D").resample(rule)
    stats = r.agg(["mean", "std", "count"])
    sem = stats["std"] / np.sqrt(stats["count"])
    lower = stats["mean"] - z * sem
    upper = stats["mean"] + z * sem
    return stats["mean"], lower, upper, stats["count"]


def bootstrap_ci(x, n_boot=500, ci=95, seed=None):
    x = np.asarray(x.dropna())
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return x.mean(), lower, upper


def resample_with_bootstrap(series, rule="2W", n_boot=100, ci=95, seed=None):
    grouped = series.asfreq("D").resample(rule)
    means = []
    lowers = []
    uppers = []
    counts = []
    index = []
    for t, group in grouped:
        m, lo, hi = bootstrap_ci(group, n_boot=n_boot, ci=ci, seed=seed)
        means.append(m)
        lowers.append(lo)
        uppers.append(hi)
        counts.append(group.count())
        index.append(t)
    return (
        pd.Series(means, index=index),
        pd.Series(lowers, index=index),
        pd.Series(uppers, index=index),
        pd.Series(counts, index=index),
    )


def bootstrap(*args, func=np.mean, alpha=0.95, n_rep=500, seed=13):
    """
    Bootstraping confidence intervals for the mean/median value
    https://towardsdatascience.com/how-to-calculate-confidence-intervals-in-python-a8625a48e62b
    """

    np.random.seed(seed)
    data = np.asanyarray(args)
    idx = np.arange(0, data.shape[1], 1)
    sample = [
        func(*np.take(data, np.random.choice(idx, size=data.shape[1], replace=True), 1))
        for _ in range(n_rep)
    ]
    lp, m, rp = (1 - alpha) / 2, 0.5, 1 - (1 - alpha) / 2
    return np.percentile(sample, [lp * 100, m * 100, rp * 100])


def draw_lines_with_ci(
    ax,
    serie,
    label,
    marker=None,
    front_color="indigo",
    back_color="white",
    fill_color="silver",
    rule="2W",
    method="pchip",
    n_boot=100,
    ci=95,
    do_gap_dilation=False,
    seed=13,
):
    def gap_dilation(df, ignore=False):
        if ignore:
            return df
        idx = df.isna()
        return df.mask(idx.shift() | idx.shift(-1) | idx, np.nan)

    mean_v, lo_v, hi_v, n_v = resample_with_bootstrap(
        serie, seed=seed, rule=rule, n_boot=n_boot, ci=ci
    )
    norm_n = (n_v - n_v.min()) / (n_v.max() - n_v.min())
    norm_n_i = gap_dilation(norm_n, not do_gap_dilation).interpolate(method=method)
    mean_v_i = gap_dilation(mean_v, not do_gap_dilation).interpolate(method=method)
    lo_v_i = gap_dilation(lo_v, not do_gap_dilation).interpolate(method=method)
    hi_v_i = gap_dilation(hi_v, not do_gap_dilation).interpolate(method=method)
    mean_v_i.plot(ax=ax, color=front_color, marker=marker, lw=2, alpha=0.8, label=label)
    mean_v_i.plot(ax=ax, color=back_color, lw=2, alpha=1.0, label="_skip_")
    for i in range(len(mean_v_i) - 1):
        mean_v_i.iloc[i : i + 2].plot(
            ax=ax,
            color=front_color,
            alpha=0.3 + 0.7 * norm_n_i.iloc[i],
            lw=2,
            label="_skip_",
        )
    ax.fill_between(
        mean_v_i.index,
        lo_v_i,
        hi_v_i,
        color=fill_color,
        alpha=0.5,
        linewidth=0,
    )


def load_metric(metric="mean_degree", alg="DEF", directed=0, col=None):
    if alg is not None and directed is not None:
        d = "directed" if directed else "undirected"
        df = pd.read_csv(
            os.path.join(
                results_dir,
                f"{metric}.{alg}.{d}.csv",
            ),
            parse_dates=True,
            index_col=0,
        )
        if col is None:
            df.columns = [f"{i}_{alg}_{d}" for i in df.columns]
        else:
            df = df[[col]]
            df.columns = [f"{col}_{alg}_{d}"]
        return df
    else:
        return pd.read_csv(
            os.path.join(results_dir, "metrics.csv"), parse_dates=True, index_col=1
        )


def get_stars(
    p, p001=r"{\ast}{\ast}{\ast}", p01=r"{\ast}{\ast}", p05=r"{\ast}", p10="⁺", p_=""
):
    if not isinstance(p, Number):
        return p
    if p < 0.001:
        return p001
    if p < 0.010:
        return p01
    if p < 0.050:
        return p05
    if p < 0.100:
        return p10
    return p_


def calc_spearman(a, b):
    r, p = spearmanr(a, b, nan_policy="omit")
    if r > 0.99:
        r = 0.99
    return r, p


def add_spearman(a, b, ax):
    r, p = calc_spearman(a, b)
    hs, ls = ax.get_legend_handles_labels()
    rl = Line2D(
        [0],
        [0],
        marker="o",
        color="gray",
        markerfacecolor="white",
        markersize=6,
        linewidth=0.0,
    )
    return hs + [rl], ls + [
        "$r = " + f"{r:.2f}" + "^{" + get_stars(p) + "}$"  # _{Spearman}
    ]


def rescale_series(x, plus=0):
    return (x - x.min()) / (x.max() - x.min()) + plus


def fig02new(
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
    rule="Q",
    method="pchip",
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    df_left = load_metric("mean_degree", alg="DEF", directed=0).join(
        load_metric("mean_degree", alg="DEF", directed=1)
    )
    df_right = load_metric("density", alg="DEF", directed=0).join(
        load_metric("density", alg="DEF", directed=1)
    )
    fig = plt.figure(figsize=(9, 4))
    ax_left = fig.add_subplot(1, 2, 1)
    draw_lines_with_ci(
        ax_left,
        df_left["mean_degree_DEF_directed"],
        "degree (directed)",
        front_color="indigo",
        rule=rule,
        method=method,
        seed=seed,
    )
    draw_lines_with_ci(
        ax_left,
        df_left["mean_degree_DEF_undirected"],
        "degree (undirected)",
        front_color="crimson",
        rule=rule,
        method=method,
        seed=seed,
    )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_left.yaxis.set_major_formatter(xfmt)
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("average degree")
    ax_left.set_xlim(left=datetime(2019, 1, 1))
    hs, ls = add_spearman(
        df_left["mean_degree_DEF_directed"],
        df_left["mean_degree_DEF_undirected"],
        ax_left,
    )
    ax_left.legend(
        hs,
        ls,
        loc="upper right",
    )
    ax_right = fig.add_subplot(1, 2, 2)
    draw_lines_with_ci(
        ax_right,
        df_right["density_DEF_directed"],
        "density (directed)",
        front_color="indigo",
        rule=rule,
        method=method,
        seed=seed,
    )
    draw_lines_with_ci(
        ax_right,
        df_right["density_DEF_undirected"],
        "density (undirected)",
        front_color="crimson",
        rule=rule,
        method=method,
        seed=seed,
    )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_right.yaxis.set_major_formatter(xfmt)
    ax_right.set_xlabel(None)
    ax_right.set_ylabel("density")
    ax_right.set_xlim(left=datetime(2019, 1, 1))
    hs, ls = add_spearman(
        df_right["density_DEF_directed"],
        df_right["density_DEF_undirected"],
        ax_right,
    )
    ax_right.legend(
        hs,
        ls,
        loc="upper right",
    )
    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig03new_(
    results_dir=results_dir,
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
):
    def add_v(a, r, ks, ks_p):
        rl = Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            markerfacecolor="white",
            markersize=6,
            linewidth=0.0,
        )
        return [rl, rl], [
            r"$\overline{\alpha} = "
            + f"{a[1]:.3f}$ 95%CI ["
            + f"{a[0]:.3f}, {a[2]:.3f}]",
            r"$\overline{r^{2}} = "
            + f"{r[1]:.3f}$, "  # 95%CI ["+ f"{r[0]:.3f}, {r[2]:.3f}]",
            + r"$p$-$value_"
            + r"{KS} = "
            + f"{ks_p:.3f}$",
        ]

    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    with open(os.path.join(results_dir, "powerlaw.discrete.directed.pkl"), "rb") as f:
        metrics = pickle.load(f)
        fshapes = set(
            pd.read_csv(os.path.join(results_dir, "shapes_fix.csv"))[
                "date"
            ].str.replace("-", "")
        )
        metrics = {k: v for k, v in metrics.items() if k in fshapes}
    fig = plt.figure(figsize=(9, 5))
    ax_left = fig.add_subplot(1, 2, 1)
    cmap = plt.cm.Spectral(np.linspace(0, 1, len(metrics)))
    cmap[:, 3] = 0.2
    ax_left.set_prop_cycle("color", cmap)
    ax_left.set_xlim(left=0, right=3.5)
    alpha, KS_stat, KS_pvalue, r_squared = [], [], [], []
    for i, (k, v) in enumerate(metrics.items()):
        # if i > 10:
        #    break
        alpha.append(v["alpha"])
        # KS_stat.append(v["KS_stat"])
        KS_pvalue.append(v["KS_pvalue"])
        r_squared.append(v["r_squared"])
        x = np.log10(v["x_emp"])
        y_emp = np.log10(v["y_emp"])
        y_pred = np.log10(v["y_pred"])
        sns.regplot(
            ax=ax_left,
            x=x,
            y=y_emp,
            truncate=True,
            ci=False,
            scatter_kws=dict(s=1, alpha=0.02),
            line_kws=dict(lw=0, label=False),
        )
        sns.regplot(
            ax=ax_left,
            x=x,
            y=y_pred,
            order=1,
            truncate=False,
            ci=False,
            scatter_kws=dict(s=0),
            line_kws=dict(alpha=0.05, lw=2, ls="-", label=False),
        )
        # break
    alpha = bootstrap(alpha)
    KS_pvalue = np.mean(KS_pvalue)
    r_squared = bootstrap(r_squared)
    # KS_stat = bootstrap(KS_stat)
    ax_left.legend(*add_v(alpha, r_squared, KS_stat, KS_pvalue), loc="upper right")
    ax_left.set_xlabel("$log_{10}$(node degree)")
    ax_left.set_ylabel("$log_{10}$(cumulative probability)")
    plt.pcolor(np.random.rand(0, 0), cmap="Spectral")
    cbar = plt.colorbar(location="top")
    keys = list(metrics.keys())
    cbar.set_ticks(ticks=[0, 1], labels=[keys[0][:4], keys[-1][:4]])

    def add_sp(r_mean, r_ci, p_mean):
        rl = Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            markerfacecolor="white",
            markersize=6,
            linewidth=0.0,
        )
        return [rl], [
            r"$\overline{r} = "  # _{Spearman}
            + f"{r_mean:.2f}"
            + "^{"
            + get_stars(p_mean)
            + "}$ 95%CI ["
            + f"{r_ci[0]:.3f}, {r_ci[1]:.3f}]"
        ]

    with open(os.path.join(results_dir, "shared_cap.directed.pkl"), "rb") as f:
        metrics = pickle.load(f)
        fshapes = set(
            pd.read_csv(os.path.join(results_dir, "shapes_fix.csv"))[
                "date"
            ].str.replace("-", "")
        )
        metrics = {k: v for k, v in metrics.items() if k in fshapes}

    ax_right = fig.add_subplot(1, 2, 2)
    cmap = plt.cm.Spectral(np.linspace(0, 1, len(metrics)))
    cmap[:, 3] = 0.2
    ax_right.set_prop_cycle("color", cmap)
    ax_right.set_xlim(left=0, right=3.5)
    r, p = [], []
    for i, (k, v) in enumerate(metrics.items()):
        # if i > 10:
        #    break
        v = v[(v[:, 1] > 0) & (v[:, 0] > 0)]
        x = np.log10(v[:, 0])
        y = np.log10(v[:, 1])
        r_, p_ = spearmanr(x, y)
        r.append(r_)
        p.append(p_)
        sns.regplot(
            ax=ax_right,
            x=x,
            y=y,
            order=1,
            truncate=False,
            ci=False,
            scatter_kws=dict(s=1, alpha=0.01),
            line_kws=dict(alpha=0.03, lw=2, ls="-", label=False),
        )
        # break
    r = bootstrap(r)
    p_mean = np.mean(p)
    r_mean, r_ci = r[1], (r[0], r[2])
    print(f"r: {r_mean:.3f} 95%CI {r_ci}")
    print(f"p: {p_mean:.3f}")
    ax_right.legend(*add_sp(r_mean, r_ci, p_mean), loc="lower right")
    ax_right.set_xlabel("$log_{10}$(node degree)")
    ax_right.set_ylabel("$log_{10}$(shared node capacity)")
    plt.pcolor(np.random.rand(0, 0), cmap="Spectral")
    cbar = plt.colorbar(location="top")
    keys = list(metrics.keys())
    cbar.set_ticks(ticks=[0, 1], labels=[keys[0][:4], keys[-1][:4]])

    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig04new(
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
    rule="Q",
    method="pchip",
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    df = load_metric("preferential_attachment", alg=None, directed=None)
    fig = plt.figure(figsize=(9, 4))
    ax_left = fig.add_subplot(1, 2, 1)
    draw_lines_with_ci(
        ax_left,
        df["preferential_attachment"],
        "preferential attachment",
        front_color="indigo",
        rule=rule,
        method=method,
        seed=seed,
    )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_left.yaxis.set_major_formatter(xfmt)
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("preferential attachment score")
    ax_left.set_xlim(left=datetime(2019, 1, 1))
    hs, ls = ax_left.get_legend_handles_labels()
    ax_left_twinx = ax_left.twinx()
    draw_lines_with_ci(
        ax_left_twinx,
        df["preferential_attachment"] / df["mean_degree"],
        "preferential attachment (adj.)",
        front_color="crimson",
        rule=rule,
        method=method,
        seed=seed,
    )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_left_twinx.yaxis.set_major_formatter(xfmt)
    ax_left_twinx.set_xlabel(None)
    ax_left_twinx.set_ylabel("preferential attachment score (adj.)")
    ax_left_twinx.set_xlim(left=datetime(2019, 1, 1))
    h_, l_ = ax_left_twinx.get_legend_handles_labels()
    ax_left_twinx.legend(hs + h_, ls + l_, loc="upper right")
    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )

    df = load_metric("min_edge_cover", alg=None, directed=None)
    ax_right = fig.add_subplot(1, 2, 2)
    draw_lines_with_ci(
        ax_right,
        df["min_edge_cover"],
        "minimal edge cover",
        front_color="indigo",
        rule=rule,
        method=method,
        seed=seed,
    )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_right.yaxis.set_major_formatter(xfmt)
    ax_right.set_xlabel(None)
    ax_right.set_ylabel("edge cover")
    ax_right.set_xlim(left=datetime(2019, 1, 1))
    hs, ls = ax_right.get_legend_handles_labels()
    ax_right_twinx = ax_right.twinx()
    draw_lines_with_ci(
        ax_right_twinx,
        df["bridges"] / df["edges"],
        "bridges (norm.)",
        front_color="crimson",
        rule=rule,
        method=method,
        seed=seed,
    )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_right_twinx.yaxis.set_major_formatter(xfmt)
    ax_right_twinx.set_xlabel(None)
    ax_right_twinx.set_ylabel("bridges (norm.)")
    ax_right_twinx.set_xlim(left=datetime(2019, 1, 1))
    h_, l_ = add_spearman(
        df["min_edge_cover"],
        df["bridges"] / df["edges"],
        ax_right_twinx,
    )
    ax_right_twinx.legend(hs + h_, ls + l_, loc="upper left")
    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig05new(
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
    rule="Q",
    method="pchip",
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    df = load_metric("jaccard_coefficient", alg=None, directed=None)
    fig = plt.figure(figsize=(9, 4))
    ax_left = fig.add_subplot(1, 2, 1)
    draw_lines_with_ci(
        ax_left,
        df["jaccard_coefficient"],
        "Jaccard coefficient",
        front_color="indigo",
        rule=rule,
        method=method,
        seed=seed,
    )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_left.yaxis.set_major_formatter(xfmt)
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("Jaccard coefficient")
    ax_left.set_xlim(left=datetime(2019, 1, 1))
    hs, ls = ax_left.get_legend_handles_labels()
    ax_left_twinx = ax_left.twinx()
    draw_lines_with_ci(
        ax_left_twinx,
        df["resource_allocation_index"],
        "resource allocation index",
        front_color="crimson",
        rule=rule,
        method=method,
        seed=seed,
    )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_left_twinx.yaxis.set_major_formatter(xfmt)
    ax_left_twinx.set_xlabel(None)
    ax_left_twinx.set_ylabel("resource allocation index")
    ax_left_twinx.set_xlim(left=datetime(2019, 1, 1))
    h_, l_ = ax_left_twinx.get_legend_handles_labels()

    r, p = calc_spearman(df["jaccard_coefficient"], df["resource_allocation_index"])
    rl = Line2D(
        [0],
        [0],
        marker="o",
        color="gray",
        markerfacecolor="white",
        markersize=6,
        linewidth=0.0,
    )
    ax_left_twinx.legend(
        hs + h_ + [rl],
        ls + l_ + ["$r = " + f"{r:.2f}" + "^{" + get_stars(p) + "}$"],  # _{Spearman}
        loc="upper right",
    )

    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )

    df_right = load_metric("transitivity", alg="DEF", directed=0).join(
        load_metric("transitivity", alg="DEF", directed=1)
    )
    ax_right = fig.add_subplot(1, 2, 2)
    draw_lines_with_ci(
        ax_right,
        df_right["transitivity_DEF_directed"],
        "transitivity (directed)",
        front_color="indigo",
        rule=rule,
        method=method,
        seed=seed,
    )
    draw_lines_with_ci(
        ax_right,
        df_right["transitivity_DEF_undirected"],
        "transitivity (undirected)",
        front_color="crimson",
        rule=rule,
        method=method,
        seed=seed,
    )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_right.yaxis.set_major_formatter(xfmt)
    ax_right.set_xlabel(None)
    ax_right.set_ylabel("transitivity")
    ax_right.set_xlim(left=datetime(2019, 1, 1))
    hs, ls = add_spearman(
        df_right["transitivity_DEF_directed"],
        df_right["transitivity_DEF_undirected"],
        ax_right,
    )
    ax_right.legend(
        hs,
        ls,
        loc="upper right",
    )

    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig06new(
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
    rule="Q",
    method="pchip",
    colors={
        "DEF": "crimson",
        "CAP": "teal",
        "LND": "darkgreen",
        "CLN": "mediumblue",
        "ECL": "mediumorchid",
    },
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]

    fig = plt.figure(figsize=(9, 5.5))
    ax_left = fig.add_subplot(1, 2, 1)
    df_left = load_metric("average_clustering", alg="DEF", directed=0).join(
        load_metric("average_clustering", alg="CAP", directed=0).join(
            load_metric("average_clustering", alg="LND", directed=0).join(
                load_metric("average_clustering", alg="CLN", directed=0).join(
                    load_metric("average_clustering", alg="ECL", directed=0)
                )
            )
        )
    )
    for i, alg in enumerate(colors.keys()):
        if i > 0:
            r, p = calc_spearman(
                rescale_series(df_left["average_clustering_DEF_undirected"]),
                rescale_series(df_left[f"average_clustering_{alg}_undirected"]),
            )

        draw_lines_with_ci(
            ax_left,
            rescale_series(df_left[f"average_clustering_{alg}_undirected"]),
            "average clustering (undirected), $w_{" + alg + "}=1$"
            if i == 0
            else "~  $w_{"
            + alg
            + "}$, $r(_{DEF}, _{"
            + alg
            + "}) = "  # _{Spearman}
            + f"{r:.2f}"
            + "^{"
            + get_stars(p)
            + "}$",
            front_color=colors[alg],
            rule=rule,
            method=method,
            seed=seed,
        )
    ax_left.set_xlabel(None)
    ax_left.set_yscale("log")
    ax_left.set_ylabel("average clustering (scaled, $log_{10}$ scale)")
    ax_left.set_xlim(left=datetime(2019, 1, 1))
    ax_left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))

    ax_right = fig.add_subplot(1, 2, 2, sharey=ax_left)
    df_right = load_metric("average_clustering", alg="DEF", directed=1).join(
        load_metric("average_clustering", alg="CAP", directed=1).join(
            load_metric("average_clustering", alg="LND", directed=1).join(
                load_metric("average_clustering", alg="CLN", directed=1).join(
                    load_metric("average_clustering", alg="ECL", directed=1)
                )
            )
        )
    )
    for i, alg in enumerate(colors.keys()):
        if i > 0:
            r, p = calc_spearman(
                rescale_series(df_right["average_clustering_DEF_directed"]),
                rescale_series(df_right[f"average_clustering_{alg}_directed"]),
            )

        draw_lines_with_ci(
            ax_right,
            rescale_series(df_right[f"average_clustering_{alg}_directed"]),
            "average clustering (directed), $w_{" + alg + "}=1$"
            if i == 0
            else "~  $w_{"
            + alg
            + "}$, $r(_{DEF}, _{"
            + alg
            + "}) = "  # _{Spearman}
            + f"{r:.2f}"
            + "^{"
            + get_stars(p)
            + "}$",
            front_color=colors[alg],
            rule=rule,
            method=method,
            seed=seed,
        )
    ax_right.set_xlabel(None)
    ax_right.set_yscale("log")
    ax_right.set_xlim(left=datetime(2019, 1, 1))
    ax_right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(pad=0.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig07new(
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
    rule="Q",
    method="pchip",
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    df = load_metric("burt_effective_size", alg=None, directed=None)
    fig = plt.figure(figsize=(5, 4))
    ax_left = fig.add_subplot()
    draw_lines_with_ci(
        ax_left,
        df["effective_size"],
        "effective size",
        front_color="indigo",
        rule=rule,
        method=method,
        seed=seed,
    )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_left.yaxis.set_major_formatter(xfmt)
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("effective size")
    ax_left.set_xlim(left=datetime(2019, 1, 1))
    hs, ls = ax_left.get_legend_handles_labels()
    ax_left_twinx = ax_left.twinx()
    draw_lines_with_ci(
        ax_left_twinx,
        df["burt_effective_size"],
        "Burt's effective size",
        front_color="crimson",
        rule=rule,
        method=method,
        seed=seed,
    )
    # xfmt = ScalarFormatter(useMathText=True)
    # xfmt.set_powerlimits((0, 0))
    # ax_left_twinx.yaxis.set_major_formatter(xfmt)
    ax_left_twinx.set_xlabel(None)
    ax_left_twinx.set_ylabel("Burt's effective size")
    ax_left_twinx.set_xlim(left=datetime(2019, 1, 1))
    h_, l_ = ax_left_twinx.get_legend_handles_labels()
    ax_left_twinx.legend(hs + h_, ls + l_, loc="upper left")
    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig08new(
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
    rule="Q",
    method="pchip",
    colors={
        "DEF": "crimson",
        "CAP": "teal",
        "LND": "darkgreen",
        "CLN": "mediumblue",
        "ECL": "mediumorchid",
    },
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]

    df = load_metric("global_efficiency", alg=None, directed=None)
    fig = plt.figure(figsize=(9, 4))
    ax_left = fig.add_subplot(1, 2, 1)
    draw_lines_with_ci(
        ax_left,
        df["global_efficiency"],
        "global efficiency",
        front_color="indigo",
        rule=rule,
        method=method,
        seed=seed,
    )
    # xfmt = ScalarFormatter(useMathText=True)
    # xfmt.set_powerlimits((0, 0))
    # ax_left.yaxis.set_major_formatter(xfmt)
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("global efficiency")
    ax_left.set_xlim(left=datetime(2019, 1, 1))
    ax_left.legend(
        loc="upper right",
    )

    ax_right = fig.add_subplot(1, 2, 2)
    df_right = load_metric("information_centrality", alg="DEF", directed=0).join(
        load_metric("information_centrality", alg="CAP", directed=0).join(
            load_metric("information_centrality", alg="LND", directed=0).join(
                load_metric("information_centrality", alg="CLN", directed=0).join(
                    load_metric("information_centrality", alg="ECL", directed=0)
                )
            )
        )
    )
    for i, alg in enumerate(colors.keys()):
        if i > 0:
            r, p = calc_spearman(
                rescale_series(df_right["information_centrality_DEF_undirected"]),
                rescale_series(df_right[f"information_centrality_{alg}_undirected"]),
            )

        draw_lines_with_ci(
            ax_right,
            rescale_series(df_right[f"information_centrality_{alg}_undirected"]),
            "information centrality, $w_{" + alg + "}=1$"
            if i == 0
            else "~  $w_{"
            + alg
            + "}$, $r(_{DEF}, _{"
            + alg
            + "}) = "  # _{Spearman}
            + f"{r:.2f}"
            + "^{"
            + get_stars(p)
            + "}$",
            front_color=colors[alg],
            rule=rule,
            method=method,
            seed=seed,
        )
    ax_right.set_xlabel(None)
    ax_right.set_ylabel("information centrality (scaled)")
    ax_right.set_xlim(left=datetime(2019, 1, 1))
    ax_right.legend(
        loc="upper right",
    )

    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig09anew(
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
    rule="Q",
    method="pchip",
    colors={
        "DEF": "crimson",
        "CAP": "teal",
        "LND": "darkgreen",
        "CLN": "mediumblue",
        "ECL": "mediumorchid",
    },
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]

    fig = plt.figure(figsize=(9, 5.5))
    ax_left = fig.add_subplot(1, 2, 1)
    df_left = load_metric(
        "fast_label_propagation_communities", alg="DEF", directed=0
    ).join(
        load_metric("fast_label_propagation_communities", alg="CAP", directed=0).join(
            load_metric(
                "fast_label_propagation_communities", alg="LND", directed=0
            ).join(
                load_metric(
                    "fast_label_propagation_communities", alg="CLN", directed=0
                ).join(
                    load_metric(
                        "fast_label_propagation_communities", alg="ECL", directed=0
                    )
                )
            )
        )
    )
    for i, alg in enumerate(colors.keys()):
        if i > 0:
            r, p = calc_spearman(
                (df_left["fast_label_propagation_communities_DEF_undirected"]),
                (df_left[f"fast_label_propagation_communities_{alg}_undirected"]),
            )

        draw_lines_with_ci(
            ax_left,
            (df_left[f"fast_label_propagation_communities_{alg}_undirected"]),
            "FLP communities (undirected), $w_{" + alg + "}=1$"
            if i == 0
            else "~  $w_{"
            + alg
            + "}$, $r(_{DEF}, _{"
            + alg
            + "}) = "  # _{Spearman}
            + f"{r:.2f}"
            + "^{"
            + get_stars(p)
            + "}$",
            front_color=colors[alg],
            rule=rule,
            method=method,
            seed=seed,
        )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_left.yaxis.set_major_formatter(xfmt)
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("communities")
    ax_left.set_xlim(left=datetime(2019, 1, 1))
    ax_left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))

    ax_right = fig.add_subplot(1, 2, 2, sharey=ax_left)
    df_right = load_metric(
        "fast_label_propagation_communities", alg="DEF", directed=1
    ).join(
        load_metric("fast_label_propagation_communities", alg="CAP", directed=1).join(
            load_metric(
                "fast_label_propagation_communities", alg="LND", directed=1
            ).join(
                load_metric(
                    "fast_label_propagation_communities", alg="CLN", directed=1
                ).join(
                    load_metric(
                        "fast_label_propagation_communities", alg="ECL", directed=1
                    )
                )
            )
        )
    )
    for i, alg in enumerate(colors.keys()):
        if i > 0:
            r, p = calc_spearman(
                (df_right["fast_label_propagation_communities_DEF_directed"]),
                (df_right[f"fast_label_propagation_communities_{alg}_directed"]),
            )

        draw_lines_with_ci(
            ax_right,
            (df_right[f"fast_label_propagation_communities_{alg}_directed"]),
            "FLP communities (directed), $w_{" + alg + "}=1$"
            if i == 0
            else "~  $w_{"
            + alg
            + "}$, $r(_{DEF}, _{"
            + alg
            + "}) = "  # _{Spearman}
            + f"{r:.2f}"
            + "^{"
            + get_stars(p)
            + "}$",
            front_color=colors[alg],
            rule=rule,
            method=method,
            seed=seed,
        )
    ax_right.set_xlabel(None)
    ax_right.set_xlim(left=datetime(2019, 1, 1))
    ax_right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(pad=0.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig09bnew(
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
    rule="Q",
    method="pchip",
    colors={
        "DEF": "crimson",
        "CAP": "teal",
        "LND": "darkgreen",
        "CLN": "mediumblue",
        "ECL": "mediumorchid",
    },
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]

    fig = plt.figure(figsize=(9, 5.5))
    ax_left = fig.add_subplot(1, 2, 1)
    df_left = load_metric("greedy_modularity_communities", alg="DEF", directed=0).join(
        load_metric("greedy_modularity_communities", alg="CAP", directed=0).join(
            load_metric("greedy_modularity_communities", alg="LND", directed=0).join(
                load_metric(
                    "greedy_modularity_communities", alg="CLN", directed=0
                ).join(
                    load_metric("greedy_modularity_communities", alg="ECL", directed=0)
                )
            )
        )
    )
    for i, alg in enumerate(colors.keys()):
        if i > 0:
            r, p = calc_spearman(
                (df_left["greedy_modularity_communities_DEF_undirected"]),
                (df_left[f"greedy_modularity_communities_{alg}_undirected"]),
            )

        draw_lines_with_ci(
            ax_left,
            (df_left[f"greedy_modularity_communities_{alg}_undirected"]),
            "GM communities (undirected), $w_{" + alg + "}=1$"
            if i == 0
            else "~  $w_{"
            + alg
            + "}$, $r(_{DEF}, _{"
            + alg
            + "}) = "  # _{Spearman}
            + f"{r:.2f}"
            + "^{"
            + get_stars(p)
            + "}$",
            front_color=colors[alg],
            rule=rule,
            method=method,
            seed=seed,
        )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_left.yaxis.set_major_formatter(xfmt)
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("communities")
    ax_left.set_xlim(left=datetime(2019, 1, 1))
    ax_left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))

    ax_right = fig.add_subplot(1, 2, 2, sharey=ax_left)
    df_right = load_metric("greedy_modularity_communities", alg="DEF", directed=1).join(
        load_metric("greedy_modularity_communities", alg="CAP", directed=1).join(
            load_metric("greedy_modularity_communities", alg="LND", directed=1).join(
                load_metric(
                    "greedy_modularity_communities", alg="CLN", directed=1
                ).join(
                    load_metric("greedy_modularity_communities", alg="ECL", directed=1)
                )
            )
        )
    )
    for i, alg in enumerate(colors.keys()):
        if i > 0:
            r, p = calc_spearman(
                (df_right["greedy_modularity_communities_DEF_directed"]),
                (df_right[f"greedy_modularity_communities_{alg}_directed"]),
            )

        draw_lines_with_ci(
            ax_right,
            (df_right[f"greedy_modularity_communities_{alg}_directed"]),
            "GM communities (directed), $w_{" + alg + "}=1$"
            if i == 0
            else "~  $w_{"
            + alg
            + "}$, $r(_{DEF}, _{"
            + alg
            + "}) = "  # _{Spearman}
            + f"{r:.2f}"
            + "^{"
            + get_stars(p)
            + "}$",
            front_color=colors[alg],
            rule=rule,
            method=method,
            seed=seed,
        )
    ax_right.set_xlabel(None)
    ax_right.set_xlim(left=datetime(2019, 1, 1))
    ax_right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(pad=0.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig10anew(
    results_dir=results_dir,
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    df = pd.read_csv(
        os.path.join(results_dir, "ks_metrics.DEF.undirected.csv"),
        parse_dates=True,
        index_col=0,
    )
    fig = plt.figure(figsize=(9, 4))
    b = df["ks_p"].asfreq("D")
    n = b.dropna()
    stat_5 = (n < 0.05).sum()
    stat_1 = (n < 0.01).sum()
    print("K-S tests DEF (undirected):")
    print(f"Share of rejected ps < 0.05: {stat_5 / len(n):.3f}")
    print(f"Number of snapshots affected: {int(stat_5)} / {len(n)}")
    print(f"Share of rejected ps < 0.01: {stat_1 / len(n):.3f}")
    print(f"Number of snapshots affected: {int(stat_1)} / {len(n)}")
    i = mdates.date2num(b.index)
    ax_left = fig.add_subplot(1, 2, 1)
    sns.regplot(
        ax=ax_left,
        x=i,
        y=b,
        order=7,
        color="mediumblue",
        scatter_kws=dict(alpha=0.3),
        line_kws=dict(alpha=0.5, lw=1, ls="-.", label="undirected (approx.)"),
    )
    sns.regplot(
        ax=ax_left,
        x=i,
        y=[0.05 for i in range(len(b))],
        color="orange",
        scatter=False,
        truncate=False,
        line_kws=dict(alpha=0.8, lw=1.5, ls="--"),
        label=f"$p < 0.05$, $n={int(stat_5)}$, ${stat_5 * 100 / len(n):.1f}$%",
    )
    sns.regplot(
        ax=ax_left,
        x=i,
        y=[0.01 for i in range(len(b))],
        color="crimson",
        scatter=False,
        truncate=False,
        line_kws=dict(alpha=0.8, lw=1.5, ls="--"),
        label=f"$p < 0.01$, $n={int(stat_1)}$, ${stat_1 * 100 / len(n):.1f}$%",
    )
    ax_left.set_xlim(
        left=mdates.date2num(datetime(2019, 1, 1)),
        right=mdates.date2num(datetime(2023, 10, 1)),
    )
    ax_left.set_xticks(
        ticks=[mdates.date2num(datetime(i, 1, 1)) for i in range(2019, 2024)],
        labels=[i for i in range(2019, 2024)],
    )
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("Kolmogorov–Smirnov test $p$-value\n(degree distribution)")
    ax_left.set_ylim(bottom=-0.05, top=1.05)
    ax_left.legend(
        loc="upper right",
        bbox_to_anchor=(1, 0.95),
    )

    df = pd.read_csv(
        os.path.join(results_dir, "ks_metrics.DEF.directed.csv"),
        parse_dates=True,
        index_col=0,
    )
    ax_right = fig.add_subplot(1, 2, 2, sharey=ax_left)
    b = df["ks_p"].asfreq("D")
    n = b.dropna()
    stat_5 = (n < 0.05).sum()
    stat_1 = (n < 0.01).sum()
    print("K-S tests DEF (directed):")
    print(f"Share of rejected ps < 0.05: {stat_5 / len(n):.3f}")
    print(f"Number of snapshots affected: {int(stat_5)} / {len(n)}")
    print(f"Share of rejected ps < 0.01: {stat_1 / len(n):.3f}")
    print(f"Number of snapshots affected: {int(stat_1)} / {len(n)}")
    i = mdates.date2num(b.index)
    sns.regplot(
        ax=ax_right,
        x=i,
        y=b,
        order=7,
        color="mediumblue",
        scatter_kws=dict(alpha=0.3),
        line_kws=dict(alpha=0.5, lw=1, ls="-.", label="directed (approx.)"),
    )
    sns.regplot(
        ax=ax_right,
        x=i,
        y=[0.05 for i in range(len(b))],
        color="orange",
        scatter=False,
        truncate=False,
        line_kws=dict(alpha=0.8, lw=1.5, ls="--"),
        label=f"$p < 0.05$, $n={int(stat_5)}$, ${stat_5 * 100 / len(n):.1f}$%",
    )
    sns.regplot(
        ax=ax_right,
        x=i,
        y=[0.01 for i in range(len(b))],
        color="crimson",
        scatter=False,
        truncate=False,
        line_kws=dict(alpha=0.8, lw=1.5, ls="--"),
        label=f"$p < 0.01$, $n={int(stat_1)}$, ${stat_1 * 100 / len(n):.1f}$%",
    )
    ax_right.set_xlim(
        left=mdates.date2num(datetime(2019, 1, 1)),
        right=mdates.date2num(datetime(2023, 10, 1)),
    )
    ax_right.set_xticks(
        ticks=[mdates.date2num(datetime(i, 1, 1)) for i in range(2019, 2024)],
        labels=[i for i in range(2019, 2024)],
    )
    ax_right.set_xlabel(None)
    ax_right.set_ylabel(None)
    ax_right.set_ylim(bottom=-0.05, top=1.05)
    ax_right.legend(
        loc="upper right",
        bbox_to_anchor=(1, 0.95),
    )
    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig10bnew(
    results_dir=results_dir,
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    df = pd.read_csv(
        os.path.join(results_dir, "ks_metrics.CAP.undirected.csv"),
        parse_dates=True,
        index_col=0,
    )
    fig = plt.figure(figsize=(9, 4))
    b = df["ks_p"].asfreq("D")
    n = b.dropna()
    stat_5 = (n < 0.05).sum()
    stat_1 = (n < 0.01).sum()
    print("K-S tests CAP (undirected):")
    print(f"Share of rejected ps < 0.05: {stat_5 / len(n):.3f}")
    print(f"Number of snapshots affected: {int(stat_5)} / {len(n)}")
    print(f"Share of rejected ps < 0.01: {stat_1 / len(n):.3f}")
    print(f"Number of snapshots affected: {int(stat_1)} / {len(n)}")
    i = mdates.date2num(b.index)
    ax_left = fig.add_subplot(1, 2, 1)
    sns.regplot(
        ax=ax_left,
        x=i,
        y=b,
        order=7,
        color="mediumblue",
        scatter_kws=dict(alpha=0.3),
        line_kws=dict(alpha=0.5, lw=1, ls="-.", label="undirected (approx.)"),
    )
    sns.regplot(
        ax=ax_left,
        x=i,
        y=[0.05 for i in range(len(b))],
        color="orange",
        scatter=False,
        truncate=False,
        line_kws=dict(alpha=0.8, lw=1.5, ls="--"),
        label=f"$p < 0.05$, $n={int(stat_5)}$, ${stat_5 * 100 / len(n):.1f}$%",
    )
    sns.regplot(
        ax=ax_left,
        x=i,
        y=[0.01 for i in range(len(b))],
        color="crimson",
        scatter=False,
        truncate=False,
        line_kws=dict(alpha=0.8, lw=1.5, ls="--"),
        label=f"$p < 0.01$, $n={int(stat_1)}$, ${stat_1 * 100 / len(n):.1f}$%",
    )
    ax_left.set_xlim(
        left=mdates.date2num(datetime(2019, 1, 1)),
        right=mdates.date2num(datetime(2023, 10, 1)),
    )
    ax_left.set_xticks(
        ticks=[mdates.date2num(datetime(i, 1, 1)) for i in range(2019, 2024)],
        labels=[i for i in range(2019, 2024)],
    )
    ax_left.set_xlabel(None)
    ax_left.set_ylabel(
        "Kolmogorov–Smirnov test $p$-value\n(shared node capacity distribution)"
    )
    ax_left.set_ylim(bottom=-0.05, top=1.05)
    ax_left.legend(
        loc="upper right",
        bbox_to_anchor=(1, 0.95),
    )

    df = pd.read_csv(
        os.path.join(results_dir, "ks_metrics.CAP.directed.csv"),
        parse_dates=True,
        index_col=0,
    )
    ax_right = fig.add_subplot(1, 2, 2, sharey=ax_left)
    b = df["ks_p"].asfreq("D")
    n = b.dropna()
    stat_5 = (n < 0.05).sum()
    stat_1 = (n < 0.01).sum()
    print("K-S tests (directed):")
    print(f"Share of rejected ps < 0.05: {stat_5 / len(n):.3f}")
    print(f"Number of snapshots affected: {int(stat_5)} / {len(n)}")
    print(f"Share of rejected ps < 0.01: {stat_1 / len(n):.3f}")
    print(f"Number of snapshots affected: {int(stat_1)} / {len(n)}")
    i = mdates.date2num(b.index)
    sns.regplot(
        ax=ax_right,
        x=i,
        y=b,
        order=7,
        color="mediumblue",
        scatter_kws=dict(alpha=0.3),
        line_kws=dict(alpha=0.5, lw=1, ls="-.", label="directed (approx.)"),
    )
    sns.regplot(
        ax=ax_right,
        x=i,
        y=[0.05 for i in range(len(b))],
        color="orange",
        scatter=False,
        truncate=False,
        line_kws=dict(alpha=0.8, lw=1.5, ls="--"),
        label=f"$p < 0.05$, $n={int(stat_5)}$, ${stat_5 * 100 / len(n):.1f}$%",
    )
    sns.regplot(
        ax=ax_right,
        x=i,
        y=[0.01 for i in range(len(b))],
        color="crimson",
        scatter=False,
        truncate=False,
        line_kws=dict(alpha=0.8, lw=1.5, ls="--"),
        label=f"$p < 0.01$, $n={int(stat_1)}$, ${stat_1 * 100 / len(n):.1f}$%",
    )
    ax_right.set_xlim(
        left=mdates.date2num(datetime(2019, 1, 1)),
        right=mdates.date2num(datetime(2023, 10, 1)),
    )
    ax_right.set_xticks(
        ticks=[mdates.date2num(datetime(i, 1, 1)) for i in range(2019, 2024)],
        labels=[i for i in range(2019, 2024)],
    )
    ax_right.set_xlabel(None)
    ax_right.set_ylabel(None)  # "Kolmogorov–Smirnov test\n$p$-value")
    ax_right.set_ylim(bottom=-0.05, top=1.05)
    ax_right.legend(
        loc="upper right",
        bbox_to_anchor=(1, 0.95),
    )
    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig11anew(
    results_dir=results_dir,
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    df = pd.read_csv(
        os.path.join(results_dir, "ks_metrics.DEF.undirected.csv"),
        parse_dates=True,
        index_col=0,
    )
    fig = plt.figure(figsize=(9, 4))
    b = df["wasserstein_distance"].asfreq("D")
    print(f"Average WD DEF (undirected): {b.dropna().mean():.3f}")
    i = mdates.date2num(b.index)
    ax_left = fig.add_subplot(1, 2, 1)
    sns.regplot(
        ax=ax_left,
        x=i,
        y=b,
        order=7,
        color="mediumblue",
        scatter_kws=dict(alpha=0.3),
        line_kws=dict(alpha=0.5, lw=1, ls="-.", label="undirected (approx.)"),
    )
    ax_left.set_xlim(
        left=mdates.date2num(datetime(2019, 1, 1)),
        right=mdates.date2num(datetime(2023, 10, 1)),
    )
    ax_left.set_xticks(
        ticks=[mdates.date2num(datetime(i, 1, 1)) for i in range(2019, 2024)],
        labels=[i for i in range(2019, 2024)],
    )
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("Wasserstein distance\n(degree distribution)")
    ax_left.set_ylim(bottom=-0.05, top=1.05)
    ax_left.legend(
        loc="upper right",
        # bbox_to_anchor=(1, 0.95),
    )

    df = pd.read_csv(
        os.path.join(results_dir, "ks_metrics.DEF.directed.csv"),
        parse_dates=True,
        index_col=0,
    )
    ax_right = fig.add_subplot(1, 2, 2, sharey=ax_left)
    b = df["wasserstein_distance"].asfreq("D")
    print(f"Average WD DEF (directed): {b.dropna().mean():.3f}")
    i = mdates.date2num(b.index)
    sns.regplot(
        ax=ax_right,
        x=i,
        y=b,
        order=7,
        color="mediumblue",
        scatter_kws=dict(alpha=0.3),
        line_kws=dict(alpha=0.5, lw=1, ls="-.", label="directed (approx.)"),
    )
    ax_right.set_xlim(
        left=mdates.date2num(datetime(2019, 1, 1)),
        right=mdates.date2num(datetime(2023, 10, 1)),
    )
    ax_right.set_xticks(
        ticks=[mdates.date2num(datetime(i, 1, 1)) for i in range(2019, 2024)],
        labels=[i for i in range(2019, 2024)],
    )
    ax_right.set_xlabel(None)
    ax_right.set_ylabel(None)
    ax_right.set_ylim(bottom=-0.05, top=1.05)
    ax_right.legend(
        loc="upper right",
        # bbox_to_anchor=(1, 0.95),
    )
    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig11bnew(
    results_dir=results_dir,
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    df = pd.read_csv(
        os.path.join(results_dir, "ks_metrics.CAP.undirected.csv"),
        parse_dates=True,
        index_col=0,
    )
    fig = plt.figure(figsize=(9, 4))
    b = df["wasserstein_distance"].asfreq("D")
    print(f"Average WD CAP (undirected): {b.dropna().mean():.3f}")
    i = mdates.date2num(b.index)
    ax_left = fig.add_subplot(1, 2, 1)
    sns.regplot(
        ax=ax_left,
        x=i,
        y=b,
        order=7,
        color="mediumblue",
        scatter_kws=dict(alpha=0.3),
        line_kws=dict(alpha=0.5, lw=1, ls="-.", label="undirected (approx.)"),
    )
    ax_left.set_xlim(
        left=mdates.date2num(datetime(2019, 1, 1)),
        right=mdates.date2num(datetime(2023, 10, 1)),
    )
    ax_left.set_xticks(
        ticks=[mdates.date2num(datetime(i, 1, 1)) for i in range(2019, 2024)],
        labels=[i for i in range(2019, 2024)],
    )
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("Wasserstein distance\n(shared node capacity distribution)")
    ax_left.set_ylim(bottom=-0.05, top=3 * 10**6)
    ax_left.legend(
        loc="upper right",
        # bbox_to_anchor=(1, 0.95),
    )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_left.yaxis.set_major_formatter(xfmt)
    df = pd.read_csv(
        os.path.join(results_dir, "ks_metrics.CAP.directed.csv"),
        parse_dates=True,
        index_col=0,
    )
    ax_right = fig.add_subplot(1, 2, 2, sharey=ax_left)
    b = df["wasserstein_distance"].asfreq("D")
    print(f"Average WD CAP (directed): {b.dropna().mean():.3f}")
    i = mdates.date2num(b.index)
    sns.regplot(
        ax=ax_right,
        x=i,
        y=b,
        order=7,
        color="mediumblue",
        scatter_kws=dict(alpha=0.3),
        line_kws=dict(alpha=0.5, lw=1, ls="-.", label="directed (approx.)"),
    )
    ax_right.set_xlim(
        left=mdates.date2num(datetime(2019, 1, 1)),
        right=mdates.date2num(datetime(2023, 10, 1)),
    )
    ax_right.set_xticks(
        ticks=[mdates.date2num(datetime(i, 1, 1)) for i in range(2019, 2024)],
        labels=[i for i in range(2019, 2024)],
    )
    ax_right.set_xlabel(None)
    ax_right.set_ylabel(None)
    ax_right.set_ylim(bottom=-0.05, top=3 * 10**6)
    ax_right.legend(
        loc="upper right",
        # bbox_to_anchor=(1, 0.95),
    )
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax_right.yaxis.set_major_formatter(xfmt)
    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig12new(
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
    rule="1Q",
    method="pchip",
):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    df = pd.read_csv(
        os.path.join(results_dir, "node_intersect_undirected.csv"),
        parse_dates=True,
        index_col=0,
    )
    fig = plt.figure(figsize=(9, 4))
    ax_left = fig.add_subplot(1, 2, 1)
    draw_lines_with_ci(
        ax_left,
        df["nodes_intersect_rate"],
        "node retention rate (undirected)",
        front_color="indigo",
        rule=rule,
        method=method,
        seed=seed,
        do_gap_dilation=True,
    )
    draw_lines_with_ci(
        ax_left,
        df["nodes_capacity_intersect_rate"],
        "shared node capacity rate (undirected)",
        front_color="crimson",
        rule=rule,
        method=method,
        seed=seed,
    )
    ax_left.set_ylim(bottom=0.85)
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("rate")
    ax_left.set_xlim(left=datetime(2019, 1, 1))
    ax_left.legend(
        *add_spearman(
            df["nodes_intersect_rate"], df["nodes_capacity_intersect_rate"], ax_left
        ),
        loc="lower left",
    )
    ax_right = fig.add_subplot(1, 2, 2, sharey=ax_left)
    df = pd.read_csv(
        os.path.join(results_dir, "node_intersect_directed.csv"),
        parse_dates=True,
        index_col=0,
    )

    draw_lines_with_ci(
        ax_right,
        df["nodes_intersect_rate"],
        "node retention rate (directed)",
        front_color="indigo",
        rule=rule,
        method=method,
        seed=seed,
        do_gap_dilation=True,
    )
    draw_lines_with_ci(
        ax_right,
        df["nodes_capacity_intersect_rate"],
        "shared node capacity rate (directed)",
        front_color="crimson",
        rule=rule,
        method=method,
        seed=seed,
    )
    ax_right.set_ylim(bottom=0.85)
    ax_right.set_xlabel(None)
    ax_right.set_ylabel(None)
    ax_right.set_xlim(left=datetime(2019, 1, 1))
    ax_right.legend(
        *add_spearman(
            df["nodes_intersect_rate"], df["nodes_capacity_intersect_rate"], ax_right
        ),
        loc="lower left",
    )

    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig14new(
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
    rule="Q",
    method="pchip",
    colors={
        "DEF": "crimson",
        "CAP": "teal",
        "LND": "darkgreen",
        "CLN": "mediumblue",
        "ECL": "mediumorchid",
    },
):
    plt.close()
    set_seed(seed)
    df_left = {}
    fname = inspect.stack()[0][3]
    fig = plt.figure(figsize=(9, 5))
    ax_left = fig.add_subplot(1, 2, 1)
    for i, alg in enumerate(colors.keys()):
        with open(
            os.path.join(results_dir, f"gini_metrics.{alg}.undirected.pkl"), "rb"
        ) as f:
            df = pickle.load(f)
        gini = [gini_coefficient(v) for v in df.values()]
        gini = pd.Series(data=gini, index=df.keys())  # .dropna()
        gini.index = pd.to_datetime(gini.index, format="%Y%m%d")
        df_left[alg] = gini
        if i > 0:
            r, p = calc_spearman(
                df_left["DEF"],
                df_left[alg],
            )
        draw_lines_with_ci(
            ax_left,
            df_left[alg],
            "Gini centrality (undirected), $w_{" + alg + "}=1$"
            if i == 0
            else "~  $w_{"
            + alg
            + "}$, $r(_{DEF}, _{"
            + alg
            + "}) = "  # _{Spearman}
            + f"{r:.2f}"
            + "^{"
            + get_stars(p)
            + "}$",
            front_color=colors[alg],
            rule=rule,
            method=method,
            seed=seed,
        )
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("Gini coefficient (betweennes centrality)")
    ax_left.set_xlim(left=datetime(2019, 1, 1))
    ax_left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07))
    df_right = {}
    ax_right = fig.add_subplot(1, 2, 2, sharey=ax_left)
    for i, alg in enumerate(colors.keys()):
        with open(
            os.path.join(results_dir, f"gini_metrics.{alg}.directed.pkl"), "rb"
        ) as f:
            df = pickle.load(f)
        gini = [gini_coefficient(v) for v in df.values()]
        gini = pd.Series(data=gini, index=df.keys())  # .dropna()
        gini.index = pd.to_datetime(gini.index, format="%Y%m%d")
        df_right[alg] = gini
        if i > 0:
            r, p = calc_spearman(
                df_right["DEF"],
                df_right[alg],
            )
        draw_lines_with_ci(
            ax_right,
            df_right[alg],
            "Gini centrality (directed), $w_{" + alg + "}=1$"
            if i == 0
            else "~  $w_{"
            + alg
            + "}$, $r(_{DEF}, _{"
            + alg
            + "}) = "  # _{Spearman}
            + f"{r:.2f}"
            + "^{"
            + get_stars(p)
            + "}$",
            front_color=colors[alg],
            rule=rule,
            method=method,
            seed=seed,
        )
    ax_right.set_xlabel(None)
    ax_right.set_ylabel(None)
    ax_right.set_xlim(left=datetime(2019, 1, 1))
    ax_right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07))

    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig13new(
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
    rule="Q",
    method="pchip",
    colors={
        "DEF": "crimson",
        "CAP": "teal",
        "LND": "darkgreen",
        "CLN": "mediumblue",
        "ECL": "mediumorchid",
    },
):
    plt.close()
    set_seed(seed)
    df_left = {}
    fname = inspect.stack()[0][3]
    fig = plt.figure(figsize=(9, 5))
    ax_left = fig.add_subplot(1, 2, 1)

    df_left = load_metric(
        "channel_intersect_cost_bootstrap",
        alg="DEF",
        directed=0,
        col="edges_intersection_rate",
    ).join(
        load_metric(
            "channel_intersect_cost_bootstrap",
            alg="CAP",
            directed=0,
            col="edges_intersection_rate",
        ).join(
            load_metric(
                "channel_intersect_cost_bootstrap",
                alg="LND",
                directed=0,
                col="edges_intersection_rate",
            ).join(
                load_metric(
                    "channel_intersect_cost_bootstrap",
                    alg="CLN",
                    directed=0,
                    col="edges_intersection_rate",
                ).join(
                    load_metric(
                        "channel_intersect_cost_bootstrap",
                        alg="ECL",
                        directed=0,
                        col="edges_intersection_rate",
                    )
                )
            )
        )
    )

    for i, alg in enumerate(colors.keys()):
        days = df_left.index.to_series().diff().dt.days.fillna(0)
        rate = df_left[f"edges_intersection_rate_{alg}_undirected"]  # .diff().fillna(0)
        r_t, p_t = calc_spearman(days, rate)
        # print(alg, r_t, p_t)

        if i > 0:
            r, p = calc_spearman(
                (df_left["edges_intersection_rate_DEF_undirected"]),
                (df_left[f"edges_intersection_rate_{alg}_undirected"]),
            )

        draw_lines_with_ci(
            ax_left,
            (df_left[f"edges_intersection_rate_{alg}_undirected"]),
            "channel retention rate (undirected),\n$w_{"
            + alg
            + "}=1$, $r(_{DEF}, \\Delta t) = "
            + f"{r_t:.2f}"
            + "^{"
            + get_stars(p_t)
            + "}$"
            if i == 0
            else "~  $w_{"
            + alg
            + "}$, $r(_{DEF}, _{"
            + alg
            + "}) = "
            + f"{r:.2f}"
            + "^{"
            + get_stars(p)
            + "}$, $r(_{"
            + alg
            + "}, \\Delta t) = "
            + f"{r_t:.2f}"
            + "^{"
            + get_stars(p_t)
            + "}$",
            front_color=colors[alg],
            rule=rule,
            method=method,
            seed=seed,
        )
    ax_left.set_xlabel(None)
    ax_left.set_ylabel("rate")
    ax_left.set_xlim(left=datetime(2019, 1, 1))
    ax_left.set_ylim(top=1.05, bottom=0.25)
    ax_left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize="small")

    ax_right = fig.add_subplot(1, 2, 2, sharey=ax_left)

    df_right = load_metric(
        "channel_intersect_cost_bootstrap",
        alg="DEF",
        directed=1,
        col="edges_intersection_rate",
    ).join(
        load_metric(
            "channel_intersect_cost_bootstrap",
            alg="CAP",
            directed=1,
            col="edges_intersection_rate",
        ).join(
            load_metric(
                "channel_intersect_cost_bootstrap",
                alg="LND",
                directed=1,
                col="edges_intersection_rate",
            ).join(
                load_metric(
                    "channel_intersect_cost_bootstrap",
                    alg="CLN",
                    directed=1,
                    col="edges_intersection_rate",
                ).join(
                    load_metric(
                        "channel_intersect_cost_bootstrap",
                        alg="ECL",
                        directed=1,
                        col="edges_intersection_rate",
                    )
                )
            )
        )
    )
    # print(df_right)
    for i, alg in enumerate(colors.keys()):
        days = df_right.index.to_series().diff().dt.days.fillna(0)
        rate = df_right[f"edges_intersection_rate_{alg}_directed"]  # .diff().fillna(0)
        r_t, p_t = calc_spearman(days, rate)
        # print(alg, r_t, p_t)

        if i > 0:
            r, p = calc_spearman(
                (df_right["edges_intersection_rate_DEF_directed"]),
                (df_right[f"edges_intersection_rate_{alg}_directed"]),
            )

        draw_lines_with_ci(
            ax_right,
            (df_right[f"edges_intersection_rate_{alg}_directed"]),
            "channel retention rate (directed),\n$w_{"
            + alg
            + "}=1$, $r(_{DEF}, \\Delta t) = "
            + f"{r_t:.2f}"
            + "^{"
            + get_stars(p_t)
            + "}$"
            if i == 0
            else "~  $w_{"
            + alg
            + "}$, $r(_{DEF}, _{"
            + alg
            + "}) = "
            + f"{r:.2f}"
            + "^{"
            + get_stars(p)
            + "}$, $r(_{"
            + alg
            + "}, \\Delta t) = "
            + f"{r_t:.2f}"
            + "^{"
            + get_stars(p_t)
            + "}$",
            front_color=colors[alg],
            rule=rule,
            method=method,
            seed=seed,
        )
    ax_right.set_xlabel(None)
    ax_right.set_ylabel("rate")
    ax_right.set_xlim(left=datetime(2019, 1, 1))
    ax_right.set_ylim(top=1.05, bottom=0.25)
    ax_right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize="small")

    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig15new(
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
    rule="Q",
    method="pchip",
    colors={
        "DEF": "crimson",
        "CAP": "teal",
        "LND": "darkgreen",
        "CLN": "mediumblue",
        "ECL": "mediumorchid",
    },
):
    plt.close()
    set_seed(seed)
    df_left = {}
    fname = inspect.stack()[0][3]
    fig = plt.figure(figsize=(9, 4))
    ax_left = fig.add_subplot(1, 2, 1)

    df_left = load_metric(
        "channel_intersect_cost_bootstrap",
        alg="DEF",
        directed=0,
        col="edges_intersection_rate",
    ).join(
        load_metric(
            "channel_intersect_cost_bootstrap",
            alg="CAP",
            directed=0,
            col="edges_intersection_rate",
        ).join(
            load_metric(
                "channel_intersect_cost_bootstrap",
                alg="LND",
                directed=0,
                col="edges_intersection_rate",
            ).join(
                load_metric(
                    "channel_intersect_cost_bootstrap",
                    alg="CLN",
                    directed=0,
                    col="edges_intersection_rate",
                ).join(
                    load_metric(
                        "channel_intersect_cost_bootstrap",
                        alg="ECL",
                        directed=0,
                        col="edges_intersection_rate",
                    )
                )
            )
        )
    )

    for i, alg in enumerate(colors.keys()):
        days = df_left.index.to_series().diff().dt.days.fillna(0) * 1
        rate = df_left[f"edges_intersection_rate_{alg}_undirected"]
        d = pd.DataFrame({"days": days, "rate": rate}).dropna()
        d = d[(d["days"] > 0) & (d["rate"] > 0)]

        sns.regplot(
            ax=ax_left,
            x=d["days"],
            y=d["rate"],
            order=1,
            truncate=False,
            robust=True,
            ci=None,
            x_jitter=0.1,
            # y_jitter=0.01,
            color=colors[alg],
            scatter_kws=dict(
                alpha=0.3,
                s=5,
            ),
            line_kws=dict(alpha=0.5, lw=2, ls="-.", label=f"{alg}"),
            seed=13,
        )
    ax_left.set_xscale("log")
    ax_left.set_xlim(left=0.5)
    ax_left.set_ylim(bottom=0.45, top=1.01)
    ax_left.set_xlabel("days between snapshots ($log_{10}$ scale)")
    ax_left.set_ylabel("rate (undirected)")
    ax_left.legend(loc="upper right", fontsize="small")  # bbox_to_anchor=(0.5, -0.07),

    ax_right = fig.add_subplot(1, 2, 2, sharey=ax_left)

    df_right = load_metric(
        "channel_intersect_cost_bootstrap",
        alg="DEF",
        directed=1,
        col="edges_intersection_rate",
    ).join(
        load_metric(
            "channel_intersect_cost_bootstrap",
            alg="CAP",
            directed=1,
            col="edges_intersection_rate",
        ).join(
            load_metric(
                "channel_intersect_cost_bootstrap",
                alg="LND",
                directed=1,
                col="edges_intersection_rate",
            ).join(
                load_metric(
                    "channel_intersect_cost_bootstrap",
                    alg="CLN",
                    directed=1,
                    col="edges_intersection_rate",
                ).join(
                    load_metric(
                        "channel_intersect_cost_bootstrap",
                        alg="ECL",
                        directed=1,
                        col="edges_intersection_rate",
                    )
                )
            )
        )
    )
    for i, alg in enumerate(colors.keys()):
        days = df_right.index.to_series().diff().dt.days.fillna(0) * 1
        rate = df_right[f"edges_intersection_rate_{alg}_directed"]
        d = pd.DataFrame({"days": days, "rate": rate}).dropna()
        d = d[(d["days"] > 0) & (d["rate"] > 0)]

        sns.regplot(
            ax=ax_right,
            x=d["days"],
            y=d["rate"],
            order=1,
            truncate=False,
            robust=True,
            ci=None,
            x_jitter=0.1,
            # y_jitter=0.01,
            color=colors[alg],
            scatter_kws=dict(
                alpha=0.3,
                s=5,
            ),
            line_kws=dict(alpha=0.5, lw=2, ls="-.", label=f"{alg}"),
            seed=13,
        )
    ax_right.set_xscale("log")
    ax_right.set_xlim(left=0.5)
    ax_right.set_ylim(bottom=0.45, top=1.01)
    ax_right.set_xlabel("days between snapshots ($log_{10}$ scale)")
    ax_right.set_ylabel("rate (directed)")
    ax_right.legend(loc="upper right", fontsize="small")  # bbox_to_anchor=(0.5, -0.07),

    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


def fig03new(
    results_dir=results_dir,
    figures_dir=figures_dir,
    show_figure=False,
    format="pdf",
    dpi=1200,
    seed=13,
):
    def add_v(a, r, ks, ks_p):
        rl = Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            markerfacecolor="white",
            markersize=6,
            linewidth=0.0,
        )
        return [rl, rl], [
            r"$\overline{|\alpha|} = "
            + f"{a[1]:.3f}$ 95%CI ["
            + f"{a[0]:.3f}, {a[2]:.3f}]",
            r"$\overline{R^{2}} = "
            + f"{r[1]:.3f}$, "  # 95%CI ["+ f"{r[0]:.3f}, {r[2]:.3f}]",
            + r"$p$-$value_"
            + r"{KS} = "
            + f"{ks_p:.3f}$",
        ]

    def add_v_(a):
        rl = Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            markerfacecolor="white",
            markersize=6,
            linewidth=0.0,
        )
        return [rl, rl], [
            r"$\overline{|\alpha|} = " + f"{a[1]:.3f}$"  # 95%CI ["
            # + f"{a[0]:.3f}, {a[2]:.3f}]",
            # r"$\overline{r^{2}} = "
            # + f"{r[1]:.3f}$, "  # 95%CI ["+ f"{r[0]:.3f}, {r[2]:.3f}]",
            # + r"$p$-$value_"
            # + r"{KS} = "
            # + f"{ks_p:.3f}$",
        ]

    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    fig = plt.figure(figsize=(9, 10))
    with open(os.path.join(results_dir, "powerlaw.discrete.directed.pkl"), "rb") as f:
        metrics = pickle.load(f)
        fshapes_ = set(
            pd.read_csv(os.path.join(results_dir, "shapes_fix_sample.csv"))[
                "date"
            ].str.replace("-", "")
        )
        fshapes = set(
            pd.read_csv(os.path.join(results_dir, "shapes_fix.csv"))[
                "date"
            ].str.replace("-", "")
        )
        metrics = {k: v for k, v in metrics.items() if k in fshapes}
        metrics_ = {k: v for k, v in metrics.items() if k in fshapes_}

    ax_left = fig.add_subplot(2, 2, 1)
    cmap = plt.cm.Spectral(np.linspace(0, 1, len(metrics)))
    cmap[:, 3] = 0.2
    ax_left.set_prop_cycle("color", cmap)
    # ax_left.set_xlim(left=0, right=3.5)
    ax_left.set_xlim(left=0.75, right=3.7)
    alpha, KS_stat, KS_pvalue, r_squared = [], [], [], []
    for i, (k, v) in enumerate(metrics.items()):
        # if i > 10:
        #    break
        alpha.append(v["alpha"])
        # KS_stat.append(v["KS_stat"])
        KS_pvalue.append(v["KS_pvalue"])
        r_squared.append(v["r_squared"])
        x = np.log10(v["x_emp"])
        y_emp = np.log10(v["y_emp"])
        y_pred = np.log10(v["y_pred"])
        sns.regplot(
            ax=ax_left,
            x=x,
            y=y_emp,
            truncate=True,
            ci=False,
            scatter_kws=dict(s=1, alpha=0.02),
            line_kws=dict(lw=0, label=False),
        )
        sns.regplot(
            ax=ax_left,
            x=x,
            y=y_pred,
            order=1,
            truncate=False,
            ci=False,
            scatter_kws=dict(s=0),
            line_kws=dict(alpha=0.06, lw=2, ls="-", label=False),
        )
        # break
    alpha = bootstrap(alpha)
    KS_pvalue = np.mean(KS_pvalue)
    r_squared = bootstrap(r_squared)
    # KS_stat = bootstrap(KS_stat)
    ax_left.legend(*add_v(alpha, r_squared, KS_stat, KS_pvalue), loc="upper right")
    ax_left.set_xlim(left=0.75, right=3.7)
    ax_left.set_xlabel("$log_{10}$(node degree)")
    ax_left.set_ylabel("$log_{10}$(cumulative probability)")
    plt.pcolor(np.random.rand(0, 0), cmap="Spectral")
    cbar = plt.colorbar(location="top")
    keys = list(metrics.keys())
    cbar.set_ticks(ticks=[0, 1], labels=[keys[0][:4], keys[-1][:4]])

    ax_left = fig.add_subplot(2, 2, 3)  #################
    cmap = plt.cm.Spectral(np.linspace(0, 1, len(metrics_)))
    cmap[:, 3] = 0.2
    ax_left.set_prop_cycle("color", cmap)
    ax_left.set_xlim(left=0.75, right=3.7)
    alpha = []  # , KS_stat, KS_pvalue, r_squared = [], [], [], []
    for i, (k, v) in enumerate(metrics_.items()):
        # if i > 10:
        #    break
        alpha.append(v["alpha"])
        x = np.log10(v["x_emp"])
        y_pred = np.log10(v["y_pred_pdf"])
        # y_pred = v["y_pred_pdf"]

        sns.regplot(
            ax=ax_left,
            x=x,
            y=y_pred,  # y_emp,
            truncate=True,
            ci=False,
            scatter_kws=dict(s=3, alpha=0.5),
            line_kws=dict(lw=0, label=False),
        )

    alpha = bootstrap(alpha)
    ax_left.set_xlim(left=0.75, right=3.7)
    # ax_left.set_ylim(bottom=-0.01)
    ax_left.legend(*add_v_(alpha), loc="upper right")
    ax_left.set_xlabel("$log_{10}$(node degree)")
    ax_left.set_ylabel("$log_{10}$(probability)")
    plt.pcolor(np.random.rand(0, 0), cmap="Spectral")
    cbar = plt.colorbar(location="top")
    keys = list(metrics.keys())
    cbar.set_ticks(ticks=[0, 1], labels=[keys[0][:4], keys[-1][:4]])

    def add_sp(r_mean, r_ci, p_mean):
        rl = Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            markerfacecolor="white",
            markersize=6,
            linewidth=0.0,
        )
        return [rl], [
            r"$\overline{r} = "  # _{Spearman}
            + f"{r_mean:.2f}"
            + "^{"
            + get_stars(p_mean)
            + "}$ 95%CI ["
            + f"{r_ci[0]:.3f}, {r_ci[1]:.3f}]"
        ]

    with open(os.path.join(results_dir, "shared_cap.directed.pkl"), "rb") as f:
        metrics = pickle.load(f)
        fshapes = set(
            pd.read_csv(os.path.join(results_dir, "shapes_fix.csv"))[
                "date"
            ].str.replace("-", "")
        )
        metrics = {k: v for k, v in metrics.items() if k in fshapes}

    ax_right = fig.add_subplot(2, 2, 2)
    cmap = plt.cm.Spectral(np.linspace(0, 1, len(metrics)))
    cmap[:, 3] = 0.2
    ax_right.set_prop_cycle("color", cmap)
    # ax_right.set_xlim(left=0, right=3.5)
    ax_right.set_xlim(left=0.75, right=3.7)
    r, p = [], []
    for i, (k, v) in enumerate(metrics.items()):
        # if i > 10:
        #    break
        v = v[(v[:, 1] > 0) & (v[:, 0] > 0)]
        x = np.log10(v[:, 0])
        y = np.log10(v[:, 1])
        r_, p_ = spearmanr(x, y)
        r.append(r_)
        p.append(p_)
        sns.regplot(
            ax=ax_right,
            x=x,
            y=y,
            order=1,
            truncate=False,
            ci=False,
            scatter_kws=dict(s=1, alpha=0.01),
            line_kws=dict(alpha=0.03, lw=2, ls="-", label=False),
        )
        # break
    r = bootstrap(r)
    p_mean = np.mean(p)
    r_mean, r_ci = r[1], (r[0], r[2])
    print(f"r: {r_mean:.3f} 95%CI {r_ci}")
    print(f"p: {p_mean:.3f}")
    # ax_right.set_xlim(left=0.75)
    ax_right.set_xlim(left=0.75, right=3.7)
    ax_right.legend(*add_sp(r_mean, r_ci, p_mean), loc="lower right")
    ax_right.set_xlabel("$log_{10}$(node degree)")
    ax_right.set_ylabel("$log_{10}$(shared node capacity)")
    plt.pcolor(np.random.rand(0, 0), cmap="Spectral")
    cbar = plt.colorbar(location="top")
    keys = list(metrics.keys())
    cbar.set_ticks(ticks=[0, 1], labels=[keys[0][:4], keys[-1][:4]])

    with open(os.path.join(results_dir, "powerlaw.discrete.directed.pkl"), "rb") as f:
        metrics = pickle.load(f)
        fshapes = pd.read_csv(os.path.join(results_dir, "shapes_fix_sample.csv"))
        fshapes["idx"] = fshapes["date"].str.replace("-", "")
        fshapes["d"] = pd.to_datetime(fshapes["date"])
        fshapes.set_index("idx", inplace=True)
        alphas = [
            {"date": fshapes.loc[k, "d"], "alpha": float(v["alpha"])}
            for k, v in metrics.items()
            if k in fshapes.index
        ]
        alphas = pd.DataFrame(alphas)
    ax_right = fig.add_subplot(2, 2, 4)

    print(alphas)
    sns.regplot(
        ax=ax_right,
        x=alphas.index,
        y=alphas["alpha"],
        order=1,
        truncate=False,
        ci=95,
        scatter_kws=dict(s=15, alpha=0.7),
        line_kws=dict(alpha=0.7, lw=2, ls="--", label=False),
    )
    # break
    ax_right.set_xlabel(None)
    ax_right.set_ylabel(r"$|\alpha|$")

    x_labels = [
        str(alphas["date"].loc[int(i)])[:7] if int(i) in alphas.index else ""
        for i in ax_right.get_xticks()
    ]
    ax_right.set_xticklabels(x_labels)

    fig.tight_layout(pad=1.01)
    fig.savefig(
        os.path.join(figures_dir, f"{fname}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
        format=format,
    )
    if show_figure:
        plt.show()


########################OLD############################
fig02new(show_figure=False)  # degree + density
fig03new(show_figure=False, format="png")  # power low
fig04new(show_figure=False)  # pref.attachement + edgecover + broges
fig05new(show_figure=False)  # Jaccard + res.alloc.index + transitivity
fig06new(show_figure=False)  # avg. clustering
fig07new(show_figure=False)  # eff.size + Burt ??
fig08new(show_figure=False)  # glob.eff. + inf.centrality
fig09anew(show_figure=False)  # FLP communities
fig09bnew(show_figure=False)  # GM communities
fig10anew(show_figure=False)  # KS
fig10bnew(show_figure=False)  # KS
fig11anew(show_figure=False)  # WD
fig11bnew(show_figure=False)  # WD
fig12new(show_figure=False)  # node intersect
fig13new(show_figure=False)  # channel intersect !!
fig14new(show_figure=False)  # gini
fig15new(show_figure=False)  # regression
