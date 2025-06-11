import os, sys
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations
from lntopo.common import DatasetFile
from lntopo.parser import ChannelAnnouncement, ChannelUpdate, NodeAnnouncement
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score



def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)


def read_shape(fname):
    g = nx.read_gml(fname)
    return len(g.nodes), len(g.edges)


def get_stamp(s):
    s = os.path.split(s)[1].split('.')[0]
    if '-' in s:
        s = s.split('-')[1]
    return s


def intersection(g1, g2):
    return set(g1.nodes).intersection(set(g2.nodes))


def edges_intersection_rate(g1, g2, common_nodes=None):
    if common_nodes is None:
        common_nodes = intersection(g1, g2)
    matched_edges = 0
    total_edges = 0
    for c in combinations(common_nodes, 2):
        total_edges += 1
        try: 
            if nx.shortest_path_length(g1, *c) >= nx.shortest_path_length(g2, *c):
                matched_edges += 1
        except:
            pass
    return matched_edges / total_edges


def nodes_intersection_rate(g1, g2, common_nodes=None):
    if common_nodes is None:
        common_nodes = intersection(g1, g2)
    return len(common_nodes) / len(g1.nodes)


def sample_graph(g, sampler=None):
    g = nx.convert_node_labels_to_integers(g, label_attribute='old_id')
    s = sampler.sample(g)
    mapping = {k: g.nodes[k]["old_id"] for k in s.nodes()}
    return nx.relabel_nodes(s, mapping)


def gini_coefficient(values, drop_zeros=False, tolerance=0.0000001):
    """Calculate the Gini coefficient"""
    # https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    x = np.array(list(values)).flatten() # all values are treated equally, arrays must be 1d
    x_min = np.amin(x)
    if x_min < tolerance:
        x += np.abs(x_min) # values cannot be negative
    if drop_zeros:
        x = x[x >= tolerance]
    else:
        x += tolerance # values cannot be 0
    n = x.shape[0] # number of array elements
    x = np.sort(x) # values must be sorted
    idx = np.arange(1, n + 1) # index per array element
    return np.sum((2 * idx - n  - 1) * x) / (n * np.sum(x)) # Gini coefficient


def fit_powlaw(g, copy=False):
    """Fit data to a power law with weights according to a log scale"""
    def powlaw(x, a, b) :
        return a * np.power(x, b)
    
    try:
        if copy:
            g = deepcopy(g)   
        degrees = np.array(g.degree)[:, 1].astype('float64')
        degrees = degrees[degrees > 0]
        probs = pd.Series(degrees).value_counts() / len(g.nodes)
        xdata, ydata = list(probs.index), list(probs)
        xdata_log = np.log10(xdata)
        ydata_log = np.log10(ydata)
        popt, pcov = curve_fit(powlaw, xdata_log, ydata_log)
        y_pred = powlaw(xdata_log, *popt)
        y_fitted = np.power(10, y_pred)
        r_squared = r2_score(ydata_log, y_pred)
        return *popt, r_squared, pcov, xdata, ydata, y_fitted
    except Exception as e:
        print(e)


def betweenness_centrality_values(g, copy=True, seed=13):
    try:
        if copy:
            g = deepcopy(g)
        return list(nx.betweenness_centrality(g, seed=seed).values())
    except Exception as e:
        print(e)


def gini_betweenness_centrality(g, copy=True, seed=13):
    try:
        return gini_coefficient(betweenness_centrality_values(g, copy=copy, seed=seed))
    except Exception as e:
        print(e)


def msg_counter(datafile):
    channel_updates = {}
    node_announcements = {}
    channel_announcements = 0
    dataset = DatasetFile().convert(datafile, 0, 0)
    for m in tqdm(dataset):
        if isinstance(m, ChannelAnnouncement):
            channel_announcements += 1
        elif isinstance(m, ChannelUpdate):
            ts = m.timestamp
            if ts in channel_updates:
                channel_updates[ts] += 1
            else:
                channel_updates[ts] = 1
        elif isinstance(m, NodeAnnouncement):
            ts = m.timestamp
            if ts in node_announcements:
                node_announcements[ts] += 1
            else:
                node_announcements[ts] = 1
    return node_announcements, channel_updates, {0 : channel_announcements}


def restore_graph(datafile, timestamp, verbose=True):
    """Restore reconstructs the network topology at a specific time in the past.
    Restore replays gossip messages from a dataset and reconstructs
    the network as it would have looked like at the specified
    timestamp in the past.
    """
    cutoff = timestamp - 2 * 7 * 24 * 3600
    channels = {}
    nodes = {}
    
    dataset = DatasetFile().convert(datafile, 0, 0)
    for m in tqdm(dataset, desc="Replaying gossip messages", disable=not verbose):
        if isinstance(m, ChannelAnnouncement):

            channels[f"{m.short_channel_id}/0"] = {
                "source": m.node_ids[0].hex(),
                "destination": m.node_ids[1].hex(),
                "timestamp": 0,
                "features": m.features.hex(),
            }

            channels[f"{m.short_channel_id}/1"] = {
                "source": m.node_ids[1].hex(),
                "destination": m.node_ids[0].hex(),
                "timestamp": 0,
                "features": m.features.hex(),
            }

        elif isinstance(m, ChannelUpdate):
            scid = f"{m.short_channel_id}/{m.direction}"
            chan = channels.get(scid, None)
            ts = m.timestamp

            if ts > timestamp:
                # Skip this update, it's in the future.
                continue

            if ts < cutoff:
                # Skip updates that cannot possibly keep this channel alive
                continue

            if chan is None:
                continue
                #raise ValueError(
                #    f"Could not find channel with short_channel_id {scid}"
                #)

            if chan["timestamp"] > ts:
                # Skip this update, it's outdated.
                continue

            chan["timestamp"] = ts
            chan["fee_base_msat"] = m.fee_base_msat
            chan["fee_proportional_millionths"] = m.fee_proportional_millionths
            chan["htlc_minimim_msat"] = m.htlc_minimum_msat
            if m.htlc_maximum_msat:
                chan["htlc_maximum_msat"] = m.htlc_maximum_msat
            chan["cltv_expiry_delta"] = m.cltv_expiry_delta
            
        elif isinstance(m, NodeAnnouncement):
            node_id = m.node_id.hex()

            old = nodes.get(node_id, None)
            if old is not None and old["timestamp"] > m.timestamp:
                continue

            alias = m.alias.replace(b'\x00', b'').decode('ASCII', 'ignore')
            nodes[node_id] = {
                "id": node_id,
                "timestamp": m.timestamp,
                "features": m.features.hex(),
                "rgb_color": m.rgb_color.hex(),
                "alias": alias,
                "addresses": ",".join([str(a) for a in m.addresses]),
                "out_degree": 0,
                "in_degree": 0,
            }

    # Cleanup pass: drop channels that haven't seen an update in 2 weeks
    todelete = []
    for scid, chan in tqdm(channels.items(), desc="Pruning outdated channels", disable=not verbose):
        if chan["timestamp"] < cutoff:
            todelete.append(scid)
        else:
            node = nodes.get(chan["source"], None)
            if node is None:
                continue
            else:
                node["out_degree"] += 1
            node = nodes.get(chan["destination"], None)
            if node is None:
                continue
            else:
                node["in_degree"] += 1

    for scid in todelete:
        del channels[scid]

    nodes = [n for n in nodes.values() if n["in_degree"] > 0 or n['out_degree'] > 0]

    #if len(channels) == 0:
    #    print(
    #        "ERROR: no channels are left after pruning, make sure to select a"
    #        "timestamp that is covered by the dataset."
    #    )
    #    return

    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["id"], **n)

    for scid, c in channels.items():
        g.add_edge(c["source"], c["destination"], scid=scid, **c)

    return g


def density(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return nx.density(g)
    except Exception as e:
        print(e)


def bridges(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return len(list(nx.bridges(g)))
    except Exception as e:
        print(e)


def min_edge_cover(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return len(list(nx.min_edge_cover(g)))
    except Exception as e:
        print(e)


def transitivity(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return nx.transitivity(g)
    except Exception as e:
        print(e)


def average_clustering(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return nx.average_clustering(g)
    except Exception as e:
        print(e)


def degree_assortativity(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return nx.degree_assortativity_coefficient(g)
    except Exception as e:
        print(e)


def global_efficiency(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return nx.global_efficiency(g)
    except Exception as e:
        print(e)


def mean_degree(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return np.mean(list(dict(g.degree).values()))
    except Exception as e:
        print(e)


def average_node_connectivity(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return nx.average_node_connectivity(g)
    except Exception as e:
        print(e)


def mean_betweenness_centrality(g, copy=True, seed=13):
    try:
        if copy:
            g = deepcopy(g)
        return np.mean(list(nx.betweenness_centrality(g, seed=seed).values()))
    except Exception as e:
        print(e)


def resource_allocation_index(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return np.mean(np.array(list(nx.resource_allocation_index(g)))[:, 2].astype('float64'))
    except Exception as e:
        print(e)


def jaccard_coefficient(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return np.mean(np.array(list(nx.jaccard_coefficient(g)))[:, 2].astype('float64'))
    except Exception as e:
        print(e)


def preferential_attachment(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return np.mean(np.array(list(nx.preferential_attachment(g)))[:, 2].astype('float64'))
    except Exception as e:
        print(e)


def common_neighbor_centrality(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return np.mean(np.array(list(nx.common_neighbor_centrality(g)))[:, 2].astype('float64'))
    except Exception as e:
        print(e)


def wiener_index(g, copy=True, use_first_component=True):
    try:
        if copy:
            g = deepcopy(g)
        if use_first_component:
            components = [g.subgraph(c) 
                            for c in sorted(nx.connected_components(g), 
                                    key=len, reverse=True) if len(c) > 10]
            if len(components):
                g = components[0]
        wi = nx.wiener_index(g)
        if np.isfinite(wi):
            return int(wi)
    except Exception as e:
        print(e)


def closeness_vitality(g, copy=True, normalized=True, use_first_component=True):
    try:
        if copy:
            g = deepcopy(g)
        if use_first_component:
            components = [g.subgraph(c) 
                            for c in sorted(nx.connected_components(g), 
                                    key=len, reverse=True) if len(c) > 10]
            if len(components):
                g = components[0]
        g.__networkx_cache__ = None
        wi = nx.wiener_index(g)
        #v = np.array([nx.closeness_vitality(g, node=n, wiener_index=wi) 
        #                                for n in g.nodes])
        v = np.array(list(nx.closeness_vitality(g, wiener_index=wi).values()))
        v = v[np.isfinite(v)].mean()
        if normalized:
            return v / wi # adj. by wiener_index
        return v
    except Exception as e:
        print(e)


def constraint(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return np.mean(list(nx.constraint(g).values()))
    except Exception as e:
        print(e)


def effective_size(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return np.mean(list(nx.effective_size(g).values()))
    except Exception as e:
        print(e)


def burt_effective_size(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return np.mean([v / g.degree(n) for n, v in nx.effective_size(g).items()])
    except Exception as e:
        print(e)


def information_centrality(g, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return np.mean(list(nx.information_centrality(g).values()))
    except Exception as e:
        print(e)


def communicability_betweenness_centrality(g, copy=True, use_first_component=True):
    try:
        if copy:
            g = deepcopy(g)
        if use_first_component:
            components = [g.subgraph(c) 
                            for c in sorted(nx.connected_components(g), 
                                    key=len, reverse=True) if len(c) > 10]
            if len(components):
                g = components[0]
        return np.mean(list(nx.communicability_betweenness_centrality(g).values()))
    except Exception as e:
        print(e)


def lpa_communities(g, seed=13, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return len(list(nx.community.asyn_lpa_communities(g, seed=seed)))
    except Exception as e:
            print(e)


def label_communities(g, seed=13, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return len(list(nx.community.fast_label_propagation_communities(g, seed=seed)))
    except Exception as e:
        print(e)


def non_randomness(g, k=None, copy=True):
    try:
        if copy:
            g = deepcopy(g)
        return nx.non_randomness(g, k=k)
    except Exception as e:
        print(e)
        return None, None


def shortest_path_length_approximation(g, n_samples=10000, copy=True):
    result = []
    if copy:
        g = deepcopy(g)
    for c in nx.connected_components(g):
            c = g.subgraph(c)
            n = c.nodes()
            lengths = []
            for _ in range(n_samples):
                try:
                    n1, n2 = random.choices(list(n), k=2)
                    length = nx.shortest_path_length(c, source=n1, target=n2)
                    lengths.append(length)
                except:
                    pass
            result.append(np.mean(lengths))
    return np.max(result)


def diameter_approximation(g, copy=True, seed=13):
    try:
        if copy:
            g = deepcopy(g)
        return nx.approximation.diameter(g, seed=seed)
    except Exception as e:
        print(e)
    return np.max([np.max(j.values()) for i, j in nx.shortest_path_length(g)])
