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
from scipy.interpolate import make_interp_spline
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from utils import gini_coefficient, set_seed, gap_dilation



if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--show', default=1, type=int)
        parser.add_argument('--data_dir', default=None, type=str)
        parser.add_argument('--results_dir', default=None, type=str)
        parser.add_argument('--shapes_file', default="shapes_fix.csv", type=str)
        args = parser.parse_args()
else:
     sys.exit()

base_dir = os.path.dirname(__file__)
if 'app' in base_dir:
    base_dir = './'
data_dir = os.path.join(base_dir, "data") if args.data_dir is None else args.data_dir
results_dir = os.path.join(base_dir, "results") if args.results_dir is None else args.results_dir
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print('data_dir:', data_dir)
print('results_dir:', results_dir)


def fig01(results_dir=results_dir, figures_dir=figures_dir,
          show_figure=False, format='pdf', dpi=1200, seed=13):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    fs = pd.read_csv(os.path.join(results_dir, args.shapes_file), 
            parse_dates=True, index_col=0)[['nodes']]
    fs = fs.asfreq('D').resample('2W').max()
    fs['idx'] = fs['nodes'].apply(lambda x: 1 if pd.notna(x) else x)

    fig = plt.figure(figsize=(5.5, 2))    
    ax = fig.add_subplot()
    fs['idx'].plot(ax=ax, color='indigo', alpha=0.2, style='o', 
                   label='', markersize=10)
    ax.set_xlim(left=datetime(2019, 1, 1), 
                right=datetime(2023, 7, 23))
    ax.set_ylabel('Snapshots')
    ax.set_xlabel(None)
    ax.set_yticklabels([''])
    fig.tight_layout(pad=1.01)
    fig.savefig(os.path.join(figures_dir, f'{fname}.{format}'), 
                dpi=dpi, bbox_inches='tight', format=format)
    if show_figure:
        plt.show()


def fig02(results_dir=results_dir, figures_dir=figures_dir,
          show_figure=False, format='pdf', dpi=1200, seed=13):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    ms = pd.read_csv(os.path.join(results_dir, 'messages.csv'), 
            parse_dates=True, index_col=0, compression='zip').fillna(0)
    ms['updates'] = ms['channel_updates'] + ms['channel_announcements']
    ms = ms[['updates', 'node_announcements']]
    ms = ms.resample('1M').sum().cumsum()

    fig = plt.figure(figsize=(8.5, 4))
    ax = fig.add_subplot(1, 2, 1)
    ms[['updates']].plot(ax=ax, 
            linewidth=2.5, color='crimson', alpha=0.7)
    ms[['node_announcements']].plot(ax=ax, 
            linewidth=2.5, color='indigo', alpha=0.7)
    ax.legend(['channel announcements\nand updates',
               'node announcements\nand updates'],
               loc='lower right')
    ax.set_xlim(left=datetime(2019, 1, 1))
    ax.set_ylabel('cumulative number ($log$ scale)')
    ax.set_xlabel(None)
    ax.set_ylim(bottom=500, top=ms.max(1).max() * 5)
    ax.set_yscale('log')

    ax = fig.add_subplot(1, 2, 2)
    bv = pd.read_csv(os.path.join(data_dir, 'bvisuals.csv'), 
                        parse_dates=True, index_col=0, sep=';')
    mp = pd.read_csv(os.path.join(data_dir, 'mempool.csv'), 
                        parse_dates=True, index_col=0, sep=';')
    fs = pd.read_csv(os.path.join(results_dir, 'shapes_fix.csv'), 
                        parse_dates=True, index_col=0).fillna(0)\
                        [['edges', 'nodes']]                   
    fs = fs.asfreq('D').resample('2W').mean().interpolate()
    fs[['nodes']].plot(ax=ax, linewidth=0)
    bv.plot(ax=ax, 
            style='o', lw=0.3, color='skyblue', alpha=0.3)
    mp.plot(ax=ax, 
            style='o', lw=0.3, color='darksalmon', alpha=0.3)
    fs[['edges']].plot(ax=ax, 
            linewidth=2.5, color='crimson', alpha=0.7)
    fs[['nodes']].plot(ax=ax, 
            linewidth=2.5, color='indigo', alpha=0.7)

    ax.set_xlim(left=datetime(2019, 1, 1), 
                right=datetime(2023, 7, 23))
    ax.set_ylabel('number ($log$ scale)')
    ax.set_xlabel(None)
    ax.set_ylim(bottom=900, top=fs.max(1).max() * 2)
    ax.set_yscale('log')

    h, l = ax.get_legend_handles_labels()
    ax.legend([h[6], h[7], h[1], h[5], ],
              ['channels', l[7], 
               '$BV$', '$MP$', ],
               loc='lower right')
    fig.tight_layout(pad=1.01)
    fig.savefig(os.path.join(figures_dir, f'{fname}.{format}'), 
                dpi=dpi, bbox_inches='tight', format=format)
    if show_figure:
        plt.show()


def fig03(results_dir=results_dir, figures_dir=figures_dir,
          show_figure=False, format='pdf', dpi=1200, seed=13):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    d = pd.read_csv(os.path.join(results_dir, "metrics.csv"), 
            parse_dates=True, index_col=1)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot()

    b = d['mean_degree'].asfreq('D').resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax, 
           linewidth=2, color='crimson', alpha=0.7, label='degree')
    ax.set_xlabel(None)
    ax.set_ylabel('average node degree')
    ax.set_xlim(left=datetime(2019, 1, 1))
    ax.legend(loc='lower left')

    ax = ax.twinx()
    b = d['density'].asfreq('D').resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax, 
           linewidth=2, color='indigo', alpha=0.7, label='density')
    ax.set_xlabel(None)
    ax.set_ylabel('density')
    ax.set_xlim(left=datetime(2019, 1, 1))
    ax.legend(loc='upper right')
    fig.tight_layout(pad=1.01)
    fig.savefig(os.path.join(figures_dir, f'{fname}.{format}'), 
                dpi=dpi, bbox_inches='tight', format=format)
    if show_figure:
        plt.show()


def fig04(results_dir=results_dir, figures_dir=figures_dir,
          show_figure=False, format='pdf', dpi=1200, seed=13):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    with open(os.path.join(results_dir, 'powlaw_metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)
        fshapes = set(pd.read_csv(os.path.join(results_dir, 'shapes_fix.csv'))\
                    ['date'].str.replace('-', ''))
        metrics = {k: v for k, v in metrics.items() if k in fshapes}
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    cmap = plt.cm.Spectral(np.linspace(0, 1, len(metrics)))
    cmap[:, 3] = 0.2
    ax.set_prop_cycle('color', cmap)
    for i, (k, v) in enumerate(metrics.items()):
        ax.scatter(v[4], v[5], s=2, alpha=0.03)
        arr = sorted(zip(v[4], v[6]), key=lambda pair: pair[0])
        t = np.array([x for x, _ in arr])
        p = np.array([y for _, y in arr])
        xnew = np.log10(np.linspace(t.min(), t.max(), 100))
        p = make_interp_spline(np.log10(t), np.log10(p), k=3)(xnew)
        ax.plot(np.power(10, xnew), np.power(10, p), linestyle='-', alpha=0.1)
    d = pd.DataFrame.from_dict(metrics, orient='index')[[0, 2]]

    print('Power low fitting:')
    print(f'α: {np.abs(d[0]).mean():.3f} ± {np.abs(d[0]).std():.3f}')
    print(f'αmin: {np.abs(d[0]).min():.3f}, αmax: {np.abs(d[0]).max():.3f}')
    print(f'r: {d[2].mean():.3f} ± {d[2].std():.3f}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(bottom=10**-4, top=1)
    ax.set_xlabel('node degree\n($log$ scale)')
    ax.set_ylabel('cumulative probability\n($log$ scale)')
    plt.pcolor(np.random.rand(0, 0), cmap='Spectral')
    cbar = plt.colorbar(location="top")
    keys = list(metrics.keys())
    cbar.set_ticks(ticks=[0, 1], labels=[keys[0][:4], keys[-1][:4]])
    fig.tight_layout(pad=1.01)
    fig.savefig(os.path.join(figures_dir, f'{fname}.{format}'), 
                dpi=dpi, bbox_inches='tight', format=format)
    if show_figure:
        plt.show()



def fig05(results_dir=results_dir, figures_dir=figures_dir,
          show_figure=False, format='pdf', dpi=1200, seed=13):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    d = pd.read_csv(os.path.join(results_dir, "metrics.csv"), 
            parse_dates=True, index_col=1)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot()

    b = d['preferential_attachment']\
        .asfreq('D').resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax, 
           linewidth=2, color='indigo', alpha=0.8, 
           label='preferential attachment')
    ax.set_xlabel(None)
    ax.set_ylabel('average preferential attachment score')
    ax.set_xlim(left=datetime(2019, 1, 1))
    ax.legend(loc='lower left')

    ax = ax.twinx()
    b = (d['preferential_attachment']/d['mean_degree'])\
        .asfreq('D').resample('1W').mean().interpolate(method='pchip')
    b.plot(ax=ax, 
           linewidth=2, color='crimson', alpha=0.7, 
           label='preferential attachment (adj.)')
    ax.set_xlabel(None)
    ax.set_ylabel('adjusted average preferential attachment score')
    ax.set_xlim(left=datetime(2019, 1, 1))
    ax.legend(loc='upper right')
    fig.tight_layout(pad=1.01)
    fig.savefig(os.path.join(figures_dir, f'{fname}.{format}'), 
                dpi=dpi, bbox_inches='tight', format=format)
    if show_figure:
        plt.show()


def fig06(results_dir=results_dir, figures_dir=figures_dir,
          show_figure=False, format='pdf', dpi=1200, seed=13):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    d = pd.read_csv(os.path.join(results_dir, "metrics.csv"),
                    parse_dates=True, index_col=1)

    fig = plt.figure(figsize=(9, 4))
    ax0 = fig.add_subplot(1, 2, 1)
    b = d['min_edge_cover'].asfreq('D')\
        .resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax0, 
           linewidth=2, color='indigo', alpha=0.8, label='edge cover')
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))
    ax0.yaxis.set_major_formatter(xfmt)
    ax0.set_xlabel(None)
    ax0.set_ylabel('minimal edge cover')
    ax0.set_xlim(left=datetime(2019, 1, 1))
    ax0.legend(loc='upper left')

    ax = ax0.twinx()
    b = d['bridges'].asfreq('D')\
        .resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax, 
           linewidth=2, color='mediumblue', alpha=0.7, label='bridges')
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))
    ax.yaxis.set_major_formatter(xfmt)
    ax.set_xlabel(None)
    ax.set_ylabel('bridges')
    ax.set_xlim(left=datetime(2019, 1, 1))
    ax.legend(loc='lower right')

    ax1 = fig.add_subplot(1, 2, 2)
    b = d['average_clustering'].asfreq('D')\
        .resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax1, 
           linewidth=2, color='crimson', alpha=0.7, label='clustering')
    ax1.set_xlabel(None)
    ax1.set_ylabel('clustering')
    ax1.set_xlim(left=datetime(2019, 1, 1))
    ax1.legend(loc='lower left')

    ax = ax1.twinx()
    b = d['transitivity'].asfreq('D')\
        .resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax, 
           linewidth=2, color='indigo', alpha=0.8, label='transitivity')
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))
    ax.yaxis.set_major_formatter(xfmt)
    ax.set_xlabel(None)
    ax.set_ylabel('transitivity')
    ax.set_xlim(left=datetime(2019, 1, 1))
    ax.legend(loc='upper right')
    fig.tight_layout(pad=1.01)
    fig.savefig(os.path.join(figures_dir, f'{fname}.{format}'), 
                dpi=dpi, bbox_inches='tight', format=format)
    if show_figure:
        plt.show()


def fig07(results_dir=results_dir, figures_dir=figures_dir,
          show_figure=False, format='pdf', dpi=1200, seed=13):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    d = pd.read_csv(os.path.join(results_dir, "metrics.csv"),
                    parse_dates=True, index_col=1)
    fig = plt.figure(figsize=(9, 4))
    ax0 = fig.add_subplot(1, 2, 1)

    b = d['global_efficiency'].asfreq('D')\
        .resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax0, 
           linewidth=2, color='indigo', alpha=0.8, 
           label='global efficiency')
    ax0.set_xlabel(None)
    ax0.set_ylabel('global efficiency')
    ax0.set_xlim(left=datetime(2019, 1, 1))
    ax0.legend(loc='lower left')

    ax = ax0.twinx()
    b = d['information_centrality'].asfreq('D')\
        .resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax, 
           linewidth=2, color='mediumblue', alpha=0.7, 
           label='information centrality')
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))    
    ax.yaxis.set_major_formatter(xfmt)
    ax.set_xlabel(None)
    ax.set_ylabel('information centrality')
    ax.set_xlim(left=datetime(2019, 1, 1))
    ax.legend(loc='upper right')

    ax1 = fig.add_subplot(1, 2, 2)
    b = d['burt_effective_size'].asfreq('D')\
        .resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax1, 
           linewidth=2, color='crimson', alpha=0.7, 
           label="Burt's effective size")
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))    
    ax1.yaxis.set_major_formatter(xfmt)
    ax1.set_xlabel(None)
    ax1.set_ylabel("Burt's effective size")
    ax1.set_xlim(left=datetime(2019, 1, 1))
    ax1.legend(loc='upper left')

    ax = ax1.twinx()
    b = d['effective_size'].asfreq('D')\
        .resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax, 
           linewidth=2, color='indigo', alpha=0.8, 
           label='effective size')
    ax.set_xlabel(None)
    ax.set_ylabel('effective size')
    ax.set_xlim(left=datetime(2019, 1, 1))
    ax.legend(loc='lower right')
    fig.tight_layout(pad=1.01)
    fig.savefig(os.path.join(figures_dir, f'{fname}.{format}'), 
                dpi=dpi, bbox_inches='tight', format=format)
    if show_figure:
        plt.show()


def fig08(results_dir=results_dir, figures_dir=figures_dir,
          show_figure=False, format='pdf', dpi=1200, seed=13):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    d = pd.read_csv(os.path.join(results_dir, "metrics.csv"),
                    parse_dates=True, index_col=1)
    fig = plt.figure(figsize=(9, 4))
    ax0 = fig.add_subplot(1, 2, 1)

    b = d['resource_allocation_index'].asfreq('D')\
        .resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax0, 
           linewidth=2, color='indigo', alpha=0.8, 
           label='resource allocation index')
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))    
    ax0.yaxis.set_major_formatter(xfmt)
    ax0.set_xlabel(None)
    ax0.set_ylabel('resource allocation index')
    ax0.set_xlim(left=datetime(2019, 1, 1))
    legend0 = [ax0.get_legend_handles_labels()]

    ax = ax0.twinx()
    b = d['jaccard_coefficient'].asfreq('D')\
        .resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax, 
           linewidth=2, color='mediumblue', alpha=0.7, 
           label='Jaccard coefficient')
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))    
    ax.yaxis.set_major_formatter(xfmt)
    ax.set_xlabel(None)
    ax.set_ylabel('Jaccard coefficient')
    ax.set_xlim(left=datetime(2019, 1, 1))
    legend0 += [ax.get_legend_handles_labels()]
    ax.legend(loc='upper right',
              handles=[j for i in legend0 for j in i[0]],
              labels=[j for i in legend0 for j in i[1]])

    ax1 = fig.add_subplot(1, 2, 2)
    b = d['lpa_communities'].asfreq('D')\
        .resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax1, 
           linewidth=2, color='crimson', alpha=0.7, 
           label="ALP communities")
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))    
    ax1.yaxis.set_major_formatter(xfmt)
    ax1.set_xlabel(None)
    ax1.set_ylabel("communities")
    ax1.set_xlim(left=datetime(2019, 1, 1))

    ax = ax1
    b = d['label_communities'].asfreq('D')\
        .resample('2W').mean().interpolate(method='pchip')
    b.plot(ax=ax, 
           linewidth=2, color='indigo', alpha=0.8, 
           label='FLP communities')
    ax.legend(loc='lower right')
    fig.tight_layout(pad=1.01)
    fig.savefig(os.path.join(figures_dir, f'{fname}.{format}'), 
                dpi=dpi, bbox_inches='tight', format=format)
    if show_figure:
        plt.show()


def fig09(results_dir=results_dir, figures_dir=figures_dir,
          show_figure=False, format='pdf', dpi=1200, seed=13):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    d = pd.read_csv(os.path.join(results_dir, "metrics.csv"),
                    parse_dates=True, index_col=1)
    fig = plt.figure(figsize=(9, 4))
    b = d['ks_p'].asfreq('D')

    n = b.dropna()
    stat = (n < 0.05).sum()
    print('K-S tests:')
    print(f"Share of rejected ps: {stat / len(n):.3f}")
    print(f"Number of snapshots affected: {int(stat)} / {len(n)}")

    i = mdates.date2num(b.index)
    ax0 = fig.add_subplot(1, 2, 2)
    sns.regplot(ax=ax0, x=i, y=b, order=7,
                color='mediumblue', scatter_kws=dict(alpha=0.3),
                line_kws=dict(alpha=.5, lw=1, ls='-.', label='approx.'))
    sns.regplot(ax=ax0, x=i, y=[0.05 for i in range(len(b))],
                color='orange', scatter=False, truncate=False,
                line_kws=dict(alpha=.8, lw=1.5, ls='--'),
                label='$p < 0.05$')
    sns.regplot(ax=ax0, x=i, y=[0.01 for i in range(len(b))],
                color='crimson', scatter=False, truncate=False,
                line_kws=dict(alpha=.8, lw=1.5, ls='--'),
                label='$p < 0.01$')
    ax0.set_xlim(left=mdates.date2num(datetime(2019, 1, 1)),
                right=mdates.date2num(datetime(2023, 10, 1)))
    ax0.set_xticks(ticks=[mdates.date2num(datetime(i, 1, 1)) 
                          for i in range(2019, 2024)],
                labels=[i for i in range(2019, 2024)])
    ax0.set_xlabel(None)
    ax0.set_ylabel('Kolmogorov–Smirnov test\n$p$-value')
    ax0.set_ylim(bottom=-0.05, top=1.05)
    ax0.legend(loc='upper right',)
    b = d['wasserstein_distance'].asfreq('D')

    print(f"Average WD: {b.dropna().mean():.3f}")

    i = mdates.date2num(b.index)
    ax1 = fig.add_subplot(1, 2, 1)
    sns.regplot(ax=ax1, x=i, y=b, order=7,
                color='mediumblue', scatter_kws=dict(alpha=0.3),
                line_kws=dict(alpha=.5, lw=1, ls='-.', label='approx.'))
    ax1.set_xlim(left=mdates.date2num(datetime(2019, 1, 1)),
                right=mdates.date2num(datetime(2023, 10, 1)))
    ax1.set_xticks(ticks=[mdates.date2num(datetime(i, 1, 1)) for i in range(2019, 2024)],
                labels=[i for i in range(2019, 2024)])
    ax1.set_xlabel(None)
    ax1.set_ylabel('Wasserstein distance')
    ax1.set_ylim(bottom=-0.05, top=1.05)
    ax1.legend(loc='upper right',)
    fig.tight_layout(pad=1.01)
    fig.savefig(os.path.join(figures_dir, f'{fname}.{format}'), 
                dpi=dpi, bbox_inches='tight', format=format)
    if show_figure:
        plt.show()


def fig10(results_dir=results_dir, figures_dir=figures_dir,
          show_figure=False, format='pdf', dpi=1200, seed=13):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    m = pd.read_csv(os.path.join(results_dir, 'extra_metrics.csv'), 
            parse_dates=True, index_col=0)
    em = m.asfreq('D').resample('2W').mean()
    dm = gap_dilation(em).interpolate(method='pchip')
    fig = plt.figure(figsize=(8.5, 4))

    ax = fig.add_subplot(1, 2, 1)
    em[[em.columns[0]]].plot(ax=ax, alpha=0)
    dm[["nodes_intersect_rate"]].plot(ax=ax, 
                style='--', linewidth=0.95, color='indigo', alpha=0.7)
    dm[["sampled_nodes_intersect_rate_mean"]].plot(ax=ax, 
                linewidth=2, color='indigo', alpha=0.7)
    l = dm["sampled_nodes_intersect_rate_mean"] -\
            dm["sampled_nodes_intersect_rate_std"]
    u = dm["sampled_nodes_intersect_rate_mean"] +\
            dm["sampled_nodes_intersect_rate_std"]
    ax.fill_between(dm.index, l, u, alpha=0.2, color='indigo')
    h, l = ax.get_legend_handles_labels()
    ax.legend([h[1], h[2]],
        [f"$avg$(full) ~ {m['nodes_intersect_rate'].mean():.3f}", 
        f"$avg$(sampled) ~ {m['sampled_nodes_intersect_rate_mean'].mean():.3f}±{m['sampled_nodes_intersect_rate_std'].mean():.3f}"],
        loc='lower right')
    ax.set_ylim((0.801, 1.04))
    ax.set_xlim(left=datetime(2019, 1, 1))
    ax.set_ylabel('node intersection rate')
    ax.set_xlabel(None)

    ax = fig.add_subplot(1, 2, 2)
    em[[em.columns[1]]].plot(ax=ax, alpha=0)
    dm[["edges_intersect_rate"]].plot(ax=ax, 
                style='--', linewidth=0.95, color='crimson', alpha=0.7)
    dm[["sampled_edges_intersect_rate_mean"]].plot(ax=ax, 
                linewidth=2, color='crimson', alpha=0.7)
    l = dm["sampled_edges_intersect_rate_mean"] -\
            dm["sampled_edges_intersect_rate_std"]
    u = dm["sampled_edges_intersect_rate_mean"] +\
            dm["sampled_edges_intersect_rate_std"]
    ax.fill_between(dm.index, l, u, alpha=0.2, color='crimson')
    h, l = ax.get_legend_handles_labels()
    ax.legend([h[1], h[2]],
        [f"$avg$(full) ~ {m['edges_intersect_rate'].mean():.3f}", 
        f"$avg$(sampled) ~ {m['sampled_edges_intersect_rate_mean'].mean():.3f}±{m['sampled_edges_intersect_rate_std'].mean():.3f}"],
        loc='lower right')
    ax.set_ylim((0.801, 1.04))
    ax.set_xlim(left=datetime(2019, 1, 1))
    ax.set_ylabel('channel intersection rate')
    ax.set_xlabel(None)
    fig.tight_layout(pad=1.01)
    fig.savefig(os.path.join(figures_dir, f'{fname}.{format}'), 
                dpi=dpi, bbox_inches='tight', format=format)
    if show_figure:
        plt.show()


def fig11(results_dir=results_dir, figures_dir=figures_dir,
          show_figure=False, format='pdf', dpi=1200, seed=13):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    with open(os.path.join(results_dir, 'tx_metrics.pkl'), 'rb') as f:
        tx_m = pickle.load(f)
        fs = set(pd.read_csv(os.path.join(results_dir, 'shapes_fix.csv'))\
                    ['date'].str.replace('-', ''))
        tx_m = {k: v for k, v in tx_m.items() if k in fs}

    fig = plt.figure(figsize=(5, 5.5))
    ax = fig.add_subplot()
    cmap = plt.cm.Spectral(np.linspace(0, 1, len(tx_m)))
    cmap[:, 3] = 0.2
    ax.set_prop_cycle('color', cmap)
    for k, v in tx_m.items():
        v.plot(ax=ax)
    pd.DataFrame(tx_m).mean(1).rename('approx.')\
        .plot(ax=ax, color='black', style='--', legend=True)
    ax.set_xscale('log')
    ax.set_xlabel('node index\n($log$ scale)')
    ax.set_ylabel('cumulative percentage of payment hops')
    plt.pcolor(np.random.rand(0, 0), cmap='Spectral')
    cbar = plt.colorbar(location="top")
    keys = list(tx_m.keys())
    cbar.set_ticks(ticks=[0, 1], labels=[keys[0][:4], keys[-1][:4]])
    fig.tight_layout(pad=1.01)
    fig.savefig(os.path.join(figures_dir, f'{fname}.{format}'), 
                dpi=dpi, bbox_inches='tight', format=format)
    if show_figure:
        plt.show()


def fig12(results_dir=results_dir, figures_dir=figures_dir,
          show_figure=False, format='pdf', dpi=1200, seed=13):
    plt.close()
    set_seed(seed)
    fname = inspect.stack()[0][3]
    with open(os.path.join(results_dir, 'gini_metrics.pkl'), 'rb') as f:
        gini_m = pickle.load(f)
        fshapes = set(pd.read_csv(os.path.join(results_dir, 'shapes_fix.csv'))\
                        ['date'].str.replace('-', ''))
        gini_m = {k: v for k, v in gini_m.items() if k in fshapes}
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 2, 1)
    cmap = plt.cm.Spectral(np.linspace(0, 1, len(gini_m)))
    cmap[:, 3] = 0.2
    ax.set_prop_cycle('color', cmap)
    for v in gini_m.values():
            sorted_vals = np.sort(np.array(v))
            sorted_vals = sorted_vals[sorted_vals > 0.0000001]
            cumvals = np.cumsum(sorted_vals)
            cumvals = np.insert(cumvals, 0, 0)  # for a zero start
            cumvals = cumvals / cumvals[-1]  # normalize
            x = np.linspace(0, 1, len(cumvals))
            ax.plot(x, cumvals, label=None)
    ax.plot([0, 1], [0, 1], '--', color='gray', label='line of equality')
    plt.pcolor(np.random.rand(0, 0), cmap='Spectral')
    cbar = plt.colorbar(ax=ax, location="top")
    keys = list(gini_m.keys())
    cbar.set_ticks(ticks=[0, 1], labels=[keys[0][:4], keys[-1][:4]])
    ax.set_xlabel('cumulative share of nodes')
    ax.set_ylabel('cumulative share of node centrality')
    plt.grid(True)
    plt.legend()
    ax = fig.add_subplot(1, 2, 2)
    gini = [gini_coefficient(v) for v in gini_m.values()]
    gini = pd.Series(data=gini, index=gini_m.keys())
    gini.index = pd.to_datetime(gini.index, format="%Y%m%d")
    gini = gini.asfreq('D').resample('2W').mean().interpolate(method='pchip')
    gini.plot(ax=ax, color='indigo', alpha=0.7)
    ax.set_ylim(bottom=0.9, top=1.0)
    ax.set_xlim(left=datetime(2019, 1, 1))
    ax.set_ylabel('Gini coefficients for node centrality')
    ax.set_xlabel(None)
    fig.tight_layout(pad=1.01)
    fig.savefig(os.path.join(figures_dir, f'{fname}.{format}'), 
                dpi=dpi, bbox_inches='tight', format=format)
    if show_figure:
        plt.show()


fig01(show_figure=False)
fig02(show_figure=False)
fig03(show_figure=False)
fig04(show_figure=False)
fig05(show_figure=False)
fig06(show_figure=False)
fig07(show_figure=False)
fig08(show_figure=False)
fig09(show_figure=False)
fig10(show_figure=False)
fig11(show_figure=False)
fig12(show_figure=False)