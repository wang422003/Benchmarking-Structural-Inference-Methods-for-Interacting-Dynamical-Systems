## Iterative Structural Inference of Directed Graphs

This repository contains our implementation for the paper *Iterative Structural Inference of Directed Graphs* accepted by NeurIPS 2022. [proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/file/39717429762da92201a750dd03386920-Paper-Conference.pdf). 

## Requirements

- Ubuntu 22.04 (optional)
- python 3.7
- pytorch >= 1.13.1
- numpy >= 1.14.5
- scipy >= 1.1.0
- pandas >= 1.5.1
- torchinfo >= 1.7.2
- CUDA 10.0
- tqdm >= 4.64.1
- sklearn >= 0.0.post1
- matplotlib >= 3.6.2



## Run experiments

In general, following args are used to select the trajectories to be used for evaluation:

```python
parser.add_argument('--b-time-steps', type=int, default=49,
                    help='Portion of time series in data to be used in benchmarking. Min = 5, Max = 49')
parser.add_argument('--b-shuffle', action='store_true', default=False,
                    help='Shuffle the data for benchmarking.')
parser.add_argument('--b-network-type', type=str, default='',
                    help='What is the network type of the graph. Please choose from: "brain_networks", "chemical_reaction_networks_in_atmosphere", "food_webs", "gene_coexpression_networks", "gene_regulatory_networks", "intercellular_networks", "landscape_networks", "man-made_organic_reaction_networks", "reaction_networks_inside_living_organism", "social_networks", "vascular_networks".')
parser.add_argument('--b-directed', action='store_true', default=False,
                    help='Default choose trajectories from undirected graphs. Use default only when running experiments on trajectories with gene_coexpression_networks and landscape_networks.')
parser.add_argument('--b-simulation-type', type=str, default='',
                    help='Either "springs" or "netsims".')
parser.add_argument('--b-suffix', type=str, default='',
    help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1.'
         ' Or "15r1" for 15 nodes, rep 1 and noise free.')
```



Reproduce the results of iSIDG in the noise-free trajectories generated by Springs simulation, and by Brain Networks with 15 nodes, with the first repetition number:

```bash
$> python3 train.py --save-probs --b-network-type 'brain_networks' --b-directed --b-simulation-type 'springs' --b-suffix '15r1' --epochs 600 --b-shuffle
```

Reproduce the results of iSIDG in the noise-free trajectories generated by NetSims simulation, and by Brain Networks with 30 nodes, with the second repetition number:

```bash
$> python3 train.py --save-probs --b-network-type 'brain_networks' --b-directed --b-simulation-type 'netsims' --b-suffix '30r2' --epochs 600 --b-shuffle
```

Reproduce the results of iSIDG in the noisy trajectories generated by NetSims simulation, and by Brain Networks with 50 nodes, with the third repetition number, with two levels of added Gaussian noise:

```bash
$> python3 train.py --save-probs --b-network-type 'brain_networks' --b-directed --b-simulation-type 'netsims' --b-suffix '50r3_n2' --epochs 600
```

Reproduce the results of iSIDG in the noise-free trajectories generated by NetSims simulation, by Brain Networks with 30 nodes, with the second repetition number, and with 5 time steps:

```bash
$> python3 train.py --save-probs --b-network-type 'brain_networks' --b-directed --b-simulation-type 'netsims' --b-suffix '30r2' --epochs 600 --b-time-steps 5 --b-shuffle
```

