import sys
import pickle
import os, argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import warnings
warnings.filterwarnings("ignore")

from utils import gini_coefficient



if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--show', default=1, type=int)
        parser.add_argument('--data_dir', default=None, type=str)
        parser.add_argument('--results_dir', default=None, type=str)
        args = parser.parse_args()
else:
     sys.exit()

base_dir = os.path.dirname(__file__)
if 'app' in base_dir:
    base_dir = './'
data_dir = os.path.join(base_dir, "data") if args.data_dir is None else args.data_dir
results_dir = os.path.join(base_dir, "results") if args.results_dir is None else args.results_dir
os.makedirs(results_dir, exist_ok=True)

print('data_dir:', data_dir)
print('results_dir:', results_dir)



def fig1(file_name='fig1.png', show_figure=False, dpi=1200):
    msgs = pd.read_csv(os.path.join(results_dir, 'messages.csv'), parse_dates=True, index_col=0, compression='zip').fillna(0)
    msgs['node_announcements\nand_updates'] = msgs['node_announcements']
    msgs['channel_announcements\nand_updates'] = msgs['channel_updates'] + msgs['channel_announcements']
    msgs.drop(['channel_announcements', 'channel_updates', 'node_announcements'], axis=1, inplace=True)
    msgs.columns = [i.replace('_', ' ') for i in msgs.columns]
    msgs = msgs.resample('1W').sum().cumsum()

    fig = plt.figure(figsize=(8, 4))

    ax = fig.add_subplot(1, 2, 1)
    #msgs.plot(ax=ax)
    msgs[['channel announcements\nand updates']].plot(ax=ax, linewidth=2, color='crimson', 
            alpha=0.7)
    msgs[['node announcements\nand updates']].plot(ax=ax, linewidth=2, color='indigo', 
            alpha=0.7)

    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax.set_ylabel('cumulative number ($log$ scale)')
    ax.set_xlabel(None)
    ax.set_ylim(bottom=500, top=msgs.max(1).max() * 5)
    ax.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), 
                                reverse=True,
                                key=lambda t: t[0]))
    ax.legend(handles, labels)
    ax = fig.add_subplot(1, 2, 2)
    bv = pd.read_csv(os.path.join(data_dir, 'bvisuals.csv'), 
                        parse_dates=True, index_col=0, sep=';')#\
    mt = pd.read_csv(os.path.join(data_dir, 'mempool.csv'), 
                        parse_dates=True, index_col=0, sep=';')#\
    mt.drop('capacity', axis=1, inplace=True)
    mt = mt[(mt['nodes'] > 1300)]
    fshapes = pd.read_csv(os.path.join(results_dir, 'shapes_fix.csv'), parse_dates=True, index_col=0)\
        .rename(columns={'edges': 'channels'}).fillna(0)
    fshapes.drop(['fname'], axis=1, inplace=True)
    fshapes = fshapes.asfreq('D').interpolate(method='pchip').resample('1W').mean().astype(int)
    fshapes\
    .rename(columns={'nodes': '$A',
                    'channels': '$A'})\
                        .plot(ax=ax, linewidth=0.01)

    bv\
    .rename(columns={'nodes': '$BV$',
                    'channels': '$A'})\
        .plot(ax=ax, style='.', lw=0.1, color='skyblue', 
            alpha=0.4)

    mt\
    .rename(columns={'nodes': '$MP$',
                    'channels': '$A'})\
        .plot(ax=ax, style='.', color='darksalmon', 
            alpha=0.4)

    fshapes[['channels']].plot(ax=ax, linewidth=2, color='crimson', 
            alpha=0.7)
    fshapes[['nodes']].plot(ax=ax, linewidth=2, color='indigo', 
            alpha=0.7)

    ax.set_xlim(left=datetime(2019, 1, 1), right=datetime(2023, 7, 23))
    ax.set_ylabel('number ($log$ scale)')
    ax.set_xlabel(None)
    ax.set_ylim(bottom=500, top=fshapes.max(1).max() * 2)
    ax.set_yscale('log')

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), 
                                reverse=True,
                                key=lambda t: t[0]))
    ax.legend(handles[:4], labels[:4])

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, file_name), dpi=dpi, bbox_inches='tight')
    if show_figure:
        plt.show()


def fig2(file_name='fig2.png', show_figure=False, dpi=1200):
    _mextra = pd.read_csv(os.path.join(results_dir, 'extra_metrics.csv'), parse_dates=True, index_col=0)
    mextra = _mextra.asfreq('D').interpolate(method='pchip').resample('1W').mean()

    fig = plt.figure(figsize=(8.5, 4))

    ax = fig.add_subplot(1, 2, 1)
    mextra[["nodes_intersect_rate"]].plot(ax=ax, style='--', linewidth=0.95, color='indigo', alpha=0.7)
    mextra[["sampled_nodes_intersect_rate_mean"]].plot(ax=ax, 
                                                       linewidth=2, color='indigo', alpha=0.7)
    lower = mextra["sampled_nodes_intersect_rate_mean"] - mextra["sampled_nodes_intersect_rate_std"]
    upper = mextra["sampled_nodes_intersect_rate_mean"] + mextra["sampled_nodes_intersect_rate_std"]
    ax.fill_between(mextra.index, lower, upper, alpha=0.2, color='indigo')

    ax.legend([f"$avg$(full) ~ {_mextra['nodes_intersect_rate'].mean():.3f}", 
            f"$avg$(sampled) ~ {_mextra['sampled_nodes_intersect_rate_mean'].mean():.3f}±{_mextra['sampled_nodes_intersect_rate_std'].mean():.3f}"])
    ax.set_ylim((0.48, 1.05))
    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax.set_ylabel('node intersection rate')
    ax.set_xlabel(None)

    ax = fig.add_subplot(1, 2, 2)
    mextra[["edges_intersect_rate"]].plot(ax=ax, style='--', linewidth=0.95, color='crimson', alpha=0.7)
    mextra[["sampled_edges_intersect_rate_mean"]].plot(ax=ax, 
                                                       linewidth=2, color='crimson', alpha=0.7)
    lower = mextra["sampled_edges_intersect_rate_mean"] - mextra["sampled_edges_intersect_rate_std"]
    upper = mextra["sampled_edges_intersect_rate_mean"] + mextra["sampled_edges_intersect_rate_std"]
    ax.fill_between(mextra.index, lower, upper, alpha=0.2, color='crimson')

    ax.legend([f"$avg$(full) ~ {_mextra['edges_intersect_rate'].mean():.3f}", 
            f"$avg$(sampled) ~ {_mextra['sampled_edges_intersect_rate_mean'].mean():.3f}±{_mextra['sampled_edges_intersect_rate_std'].mean():.3f}"])
    ax.set_ylim((0.48, 1.05))
    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax.set_ylabel('channel intersection rate')
    ax.set_xlabel(None)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, file_name), dpi=dpi, bbox_inches='tight')
    if show_figure:
        plt.show()


def fig3(file_name='fig3.png', show_figure=False, dpi=1200):
    with open(os.path.join(results_dir, 'tx_metrics.pkl'), 'rb') as f:
        tx_m = pickle.load(f)
        fshapes = set(pd.read_csv(os.path.join(results_dir, 'shapes_fix.csv'))\
                    ['date'].str.replace('-', ''))
        tx_m = {k: v for k, v in tx_m.items() if k in fshapes}

    fig = plt.figure(figsize=(5, 5.5))
    ax = fig.add_subplot()
    cmap = plt.cm.Spectral(np.linspace(0, 1, len(tx_m)))
    cmap[:, 3] = 0.2
    ax.set_prop_cycle('color', cmap)
    for k, v in tx_m.items():
        v.plot(ax=ax)
    pd.DataFrame(tx_m).mean(1).rename('average').plot(ax=ax, color='black', style='--', legend=True)
    ax.set_xscale('log')
    ax.set_xlabel('node index\n($log$ scale)')
    ax.set_ylabel('cumulative percentage of payment hops')
    plt.pcolor(np.random.rand(0, 0), cmap='Spectral')
    cbar = plt.colorbar(location="top")
    keys = list(tx_m.keys())
    cbar.set_ticks(ticks=[0, 1], labels=[keys[0][:4], keys[-1][:4]])

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, file_name), dpi=dpi, bbox_inches='tight')
    if show_figure:
        plt.show()


def fig4(file_name='fig4.png', show_figure=False, dpi=1200):
    with open(os.path.join(results_dir, 'gini_metrics.pkl'), 'rb') as f:
        gini_m = pickle.load(f)
        fshapes = set(pd.read_csv(os.path.join(results_dir, 'shapes_fix.csv'))\
                        ['date'].str.replace('-', ''))
        gini_m = {k: v for k, v in gini_m.items() if k in fshapes}

    fig = plt.figure(figsize=(8., 4))

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
    gini = gini.asfreq('D').interpolate(method='pchip').resample('1W').mean()
    gini.plot(ax=ax, color='indigo', alpha=0.7)
    ax.set_ylim(bottom=0.9, top=1.0)
    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax.set_ylabel('$Gini$ coefficients for node centrality')
    ax.set_xlabel(None)
        
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, file_name), dpi=dpi, bbox_inches='tight')
    if show_figure:
        plt.show()


def fig5(file_name='fig5.png', show_figure=False, dpi=1200):
    from scipy.interpolate import make_interp_spline
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
    print('mean', d.mean())
    print('std', d.std())
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(bottom=10**-4, top=1)
    ax.set_xlabel('node degree\n($log$ scale)')
    ax.set_ylabel('cumulative probability\n($log$ scale)')
    plt.pcolor(np.random.rand(0, 0), cmap='Spectral')
    cbar = plt.colorbar(location="top")
    keys = list(metrics.keys())
    cbar.set_ticks(ticks=[0, 1], labels=[keys[0][:4], keys[-1][:4]])

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, file_name), dpi=dpi, bbox_inches='tight')
    if show_figure:
        plt.show()


def fig6(file_name='fig6.png', show_figure=False, dpi=1200):
    d = pd.read_csv(os.path.join(results_dir, "metrics.csv"), 
            parse_dates=True, index_col=1)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot()

    b = d['preferential_attachment']\
        .asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax, linewidth=2, color='indigo', alpha=0.8, 
           label='preferential attachment')
    ax.set_xlabel(None)
    ax.set_ylabel('average preferential attachment score')
    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax.legend(loc='lower left')

    ax = ax.twinx()
    b = (d['preferential_attachment']/d['mean_degree']).asfreq('D')\
        .interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax, linewidth=2, color='crimson', alpha=0.7, 
           label='preferential attachment (adj.)')
    ax.set_xlabel(None)
    ax.set_ylabel('adjusted average preferential attachment score')
    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, file_name), dpi=dpi, bbox_inches='tight')
    if show_figure:
        plt.show()


def fig7(file_name='fig7.png', show_figure=False, dpi=1200):
    d = pd.read_csv(os.path.join(results_dir, "metrics.csv"), 
            parse_dates=True, index_col=1)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot()

    b = d['mean_degree'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax, linewidth=2, color='crimson', alpha=0.7, 
           label='average degree')
    ax.set_xlabel(None)
    ax.set_ylabel('average node degree')
    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax.legend(loc='lower left')

    ax = ax.twinx()
    b = d['density'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax, linewidth=2, color='indigo', alpha=0.7, 
           label='density')
    ax.set_xlabel(None)
    ax.set_ylabel('density')
    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, file_name), dpi=dpi, bbox_inches='tight')
    if show_figure:
        plt.show()


def fig8(file_name='fig8.png', show_figure=False, dpi=1200):
    d = pd.read_csv(os.path.join(results_dir, "metrics.csv"),
                    parse_dates=True, index_col=1)

    fig = plt.figure(figsize=(9.5, 4))
    ax0 = fig.add_subplot(1, 2, 1)

    b = d['min_edge_cover'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax0, linewidth=2, color='indigo', alpha=0.8, label='minimal edge cover')
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))
    ax0.yaxis.set_major_formatter(xfmt)
    ax0.set_xlabel(None)
    ax0.set_ylabel('minimal edge cover')
    ax0.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax0.legend(loc='upper left')

    ax = ax0.twinx()
    b = d['bridges'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax, linewidth=2, color='mediumblue', alpha=0.7, label='bridges')
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))
    ax.yaxis.set_major_formatter(xfmt)
    ax.set_xlabel(None)
    ax.set_ylabel('bridges')
    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax.legend(loc='lower right')

    ax1 = fig.add_subplot(1, 2, 2)
    b = d['average_clustering'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax1, linewidth=2, color='crimson', alpha=0.7, label='clustering')
    ax1.set_xlabel(None)
    ax1.set_ylabel('clustering')
    ax1.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax1.legend(loc='lower left')

    ax = ax1.twinx()
    b = d['transitivity'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax, linewidth=2, color='indigo', alpha=0.8, label='transitivity')
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))
    ax.yaxis.set_major_formatter(xfmt)
    ax.set_xlabel(None)
    ax.set_ylabel('transitivity')
    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, file_name), dpi=dpi, bbox_inches='tight')
    if show_figure:
        plt.show()


def fig9(file_name='fig9.png', show_figure=False, dpi=1200):
    d = pd.read_csv(os.path.join(results_dir, "metrics.csv"),
                    parse_dates=True, index_col=1)
    fig = plt.figure(figsize=(9.5, 4))
    ax0 = fig.add_subplot(1, 2, 1)

    b = d['global_efficiency'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax0, linewidth=2, color='indigo', alpha=0.8, label='global efficiency')
    ax0.set_xlabel(None)
    ax0.set_ylabel('global efficiency')
    ax0.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax0.legend(loc='lower left')

    ax = ax0.twinx()
    b = d['information_centrality'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax, linewidth=2, color='mediumblue', alpha=0.7, label='information centrality')
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))    
    ax.yaxis.set_major_formatter(xfmt)
    ax.set_xlabel(None)
    ax.set_ylabel('information centrality')
    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax.legend(loc='upper right')

    ax1 = fig.add_subplot(1, 2, 2)
    b = d['burt_effective_size'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax1, linewidth=2, color='crimson', alpha=0.7, label="Burt's effective size")
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))    
    ax1.yaxis.set_major_formatter(xfmt)
    ax1.set_xlabel(None)
    ax1.set_ylabel("Burt's effective size")
    ax1.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax1.legend(loc='upper left')

    ax = ax1.twinx()
    b = d['effective_size'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax, linewidth=2, color='indigo', alpha=0.8, label='effective size')
    ax.set_xlabel(None)
    ax.set_ylabel('effective size')
    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    ax.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, file_name), dpi=dpi, bbox_inches='tight')
    if show_figure:
        plt.show()


def fig10(file_name='fig10.png', show_figure=False, dpi=1200):
    d = pd.read_csv(os.path.join(results_dir, "metrics.csv"),
                    parse_dates=True, index_col=1)
    fig = plt.figure(figsize=(9.5, 4))
    ax0 = fig.add_subplot(1, 2, 1)

    b = d['resource_allocation_index'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax0, linewidth=2, color='indigo', alpha=0.8, label='resource allocation index')
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))    
    ax0.yaxis.set_major_formatter(xfmt)
    ax0.set_xlabel(None)
    ax0.set_ylabel('resource allocation index')
    ax0.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    legend0 = [ax0.get_legend_handles_labels()]

    ax = ax0.twinx()
    b = d['jaccard_coefficient'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax, linewidth=2, color='mediumblue', alpha=0.7, label='Jaccard coefficient')
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))    
    ax.yaxis.set_major_formatter(xfmt)
    ax.set_xlabel(None)
    ax.set_ylabel('jaccard coefficient')
    ax.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))
    legend0 += [ax.get_legend_handles_labels()]
    ax.legend(loc='upper right',
                        handles=[j for i in legend0 for j in i[0]],
                        labels=[j for i in legend0 for j in i[1]])

    ax1 = fig.add_subplot(1, 2, 2)
    b = d['lpa_communities'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax1, linewidth=2, color='crimson', alpha=0.7, label="ALP communities")
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0,0))    
    ax1.yaxis.set_major_formatter(xfmt)
    ax1.set_xlabel(None)
    ax1.set_ylabel("communities")
    ax1.set_xlim(left=datetime(2019, 1, 1))#, right=datetime(2023, 10, 1))

    ax = ax1
    b = d['label_communities'].asfreq('D').interpolate(method='pchip').resample('1W').mean()
    b.plot(ax=ax, linewidth=2, color='indigo', alpha=0.8, label='FLP communities')
    ax.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, file_name), dpi=dpi, bbox_inches='tight')
    if show_figure:
        plt.show()



fig1(show_figure=False)
fig2(show_figure=False)
fig3(show_figure=False)
fig4(show_figure=False)
fig5(show_figure=False)
fig6(show_figure=False)
fig7(show_figure=False)
fig8(show_figure=False)
fig9(show_figure=False)
fig10(show_figure=False)