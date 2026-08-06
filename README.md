## Topological and Temporal Stability Analysis of the Lightning Network

Try the Lightning Network Topology Dashboard based on this analysis: 
[ellariel.github.io/ln-data-dashboard](https://ellariel.github.io/ln-data-dashboard/)

These are the data and scripts associated with the paper on the comprehensive analysis of the network’s topology and its temporal dynamics. The folder contains several scripts and a Dockerfile used to compute various network science metrics. The final results of these calculations are available in `results/metrics.csv` and they are merged with `make_metrics.py`. All figures from the analysis paper can be reproduced using the `make_figures.py` script.

Note that since the raw data is quite large, we did not mirror it here, but it can be directly downloaded from [(Decker, 2020)](https://github.com/lnresearch/topology). Intermediate and temporary data are also not stored here, but can be reproduced with available scripts. The snapshot reconstruction scripts are in a separate repository: https://github.com/ellariel/ln-data-preparation.

The repository includes the following calculated metrics (see `scripts/utils.py`):

| **Category**                                                                          | **Metrics and attributes**                                                                                                                                                                                                                         | **Focus**                                                                          |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Network Structure** (Basic Topology Measures, Assortativity & Degree)               | *nodes*, *edges (channels)*, *components*, *density*, *diameter*, *shortest path length*, *average degree*, *degree assortativity*                                                                                                                 | Describe overall architecture, scale, and connectivity patterns of the network.    |
| **Connectivity & Resilience** (Connectivity, Clustering & Transitivity)               | *bridges*, *average node connectivity*, *minimal edge cover*, *transitivity*, *average clustering*                                                                                                                                                 | Measure cohesion, redundancy, and vulnerability to failure.                        |
| **Function & Dynamics** (Efficiency & Information Flow, Centrality, Structural Holes) | *global efficiency*, *information centrality*, *average betweenness centrality*, *communicability betweenness centrality*, *common neighbour centrality*, *constraint value*, *effective size*, *Burt's effective size*, *closeness vitality*      | Analyze information/value flow and identify nodes with control or influence roles. |
| **Emergent Patterns** (Link Prediction & Community Detection)                         | *resource allocation index*, *Jaccard coefficient*, *preferential attachment*, communities (*FLP*, *ALP*, *GM*)                                                                                                                                    | Detect underlying community structure and predict potential future links.          |
| **Other Metrics** (Topological Stability, Centrality Inequality, etc.)                | *Gini betweenness centrality*, *Wiener index*, *Wasserstein distance*, *degree and shared capacity distribution approximations*, *degree distribution entropy*, *node retention rates*, *channel retention rates*, *Kolmogorov–Smirnov statistics* | Analyze specific payment channel network features.                                 |



### Citation

```python
Valko, D., & Marx Gómez, J. (2026). Topological and temporal stability analysis of the lightning network. Applied Network Science. https://doi.org/10.1007/s41109-026-00820-4  
```

```python
@article{ValkoMarxGomez2026,
title={Topological and temporal stability analysis of the lightning network.}, 
author={Danila Valko and Jorge {Marx G\'omez}},
year={2026},
journal={Applied Network Science},
doi={10.1007/s41109-026-00820-4},
}
```


### Sources and References

- Raw data snapshots of the Lightning Network are obtained from [(Decker, 2020)](https://github.com/lnresearch/topology).
- Snapshot reconstruction scripts are in a separate repository: https://github.com/ellariel/ln-data-preparation.
- Native pathfinding algorithms are based on [[Kumble & Roos, 2021]](https://ieeexplore.ieee.org/document/9566199); [[Kumble, Epema & Roos, 2021]](https://arxiv.org/pdf/2107.10070.pdf); see also, [GitHub](https://github.com/SatwikPrabhu/Attacking-Lightning-s-anonymity).
- Lightning Network Topology Data Dashboard is available here [GitHub](https://github.com/ellariel/ln-data-dashboard)


