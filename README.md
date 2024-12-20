# :sparkles: Benchmarking Structural Inference Methods for Interacting Dynamical Systems with Synthetic Data (NeurIPS2024 Datasets and Benchmarks Track):sparkles:

<p align="center">
  <img src="website/pure_LOGO.png" alt="Project Icon" width="150"/>
</p>

[![Documentation Status](https://readthedocs.org/projects/benchmarking-structural-inference-methods/badge/?version=latest)](https://benchmarking-structural-inference-methods.readthedocs.io/en/latest/?badge=latest)
![Last Commit](https://img.shields.io/github/last-commit/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems)
[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by]. 

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey

[**Project Website**](https://structinfer.github.io/)

This repo is the official implementation of the paper 'Benchmarking Structural Inference Methods for Interacting Dynamical Systems with Synthetic Data' accepted by NeurIPS2024 Datasets and Benchmarks Track. :sparkles:

This repo maintains and updates benchmark on structural inference methods for interacting dynamical systems with synthetic data. :smile:

## Installation

Download the whole reporitory.


Different methods require different programming languages and different packages. Please refer to the README in each sub-folder (our implementation) and install the requirements:

| Methods                                                      | Paper                                                        | Official Implementation                                      | Our Implementation                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ppcor: An R Package for a Fast Calculation to Semi-partial Correlation Coefficients (ppcor) | [Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4681537/) | [Link](https://cran.r-project.org/web/packages/ppcor/index.html) | /src/models/ppcor |
| TIGRESS: Trustful Inference of Gene REgulation using Stability Selection (TIGRESS) | [Link](https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-6-145) | [Link](https://github.com/jpvert/tigress/tree/master)        | /src/models/TIGRESS |
| ARACNE: An Algorithm for the Reconstruction of Gene Regulatory Networks in a Mammalian Cellular Context (ARACNe) | [Link](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-S1-S7) | [Link](https://califano.c2b2.columbia.edu/aracne)            | /src/models/ARACNE |
| Large-Scale Mapping and Validation of Escherichia coli Transcriptional Regulation from a Compendium of Expression Profiles (CLR) | [Link](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0050008) | [Link](https://bioconductor.org/install/)                    | /src/models/CLR |
| Gene Regulatory Network Inference from Single-Cell Data Using Multivariate Information Measures (PIDC) | [Link](https://www.sciencedirect.com/science/article/pii/S2405471217303861) | [Link](https://github.com/Tchanders/NetworkInference.jl)     | /src/models/PIDC/ |
| Inferring Causal Gene Regulatory Networks from Coupled Single-Cell Expression Dynamics Using Scribe (Scribe) | [Link](https://www.sciencedirect.com/science/article/pii/S2405471220300363) | [Link](https://github.com/aristoteleo/Scribe-py/tree/master) | /src/models/scribe |
| dynGENIE3: dynamical GENIE3 for the inference of gene networks from time series expression data (dynGENIE3) | [Link](https://www.nature.com/articles/s41598-018-21715-0)   | [Link](https://github.com/vahuynh/dynGENIE3/tree/master)     | /src/models/dynGENIE3 |
| Inference of gene regulatory networks based on nonlinear ordinary differential equations (XGBGRN) | [Link](https://academic.oup.com/bioinformatics/article/36/19/4885/5709036) | [Link](https://github.com/lab319/GRNs_nonlinear_ODEs)        | /src/models/GRNs_nonlinear_ODEs |
| Neural Relational Inference for Interacting Systems (NRI)    | [Link](http://proceedings.mlr.press/v80/kipf18a/kipf18a.pdf) | [Link](http://github.com/ethanfetaya/NRI)                    | /src/models/NRI |
| Amortized Causal Discovery: Learning to Infer Causal Graphs from Time-Series Data (ACD) | [Link](https://proceedings.mlr.press/v177/lowe22a/lowe22a.pdf) | [Link](https://github.com/loeweX/AmortizedCausalDiscovery)   | /src/models/ACD |
| Neural Relational Inference with Efficient Message Passing Mechanisms (MPM) | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/16868) | [Link](https://github.com/hilbert9221/NRI-MPM)               | /src/models/MPM |
| Iterative Structural Inference of Directed Graphs (iSIDG)    | [Link](https://papers.nips.cc/paper_files/paper/2022/file/39717429762da92201a750dd03386920-Paper-Conference.pdf) | [Link](https://github.com/AoranWANGRalf/iSIDG)               | /src/models/iSIDG |
| Effective and Efficient Structural Inference with Reservoir Computing (RCSI)    | [Link](https://proceedings.mlr.press/v202/wang23ak/wang23ak.pdf) | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/RCSI)               | /src/models/RCSI  |



## Dataset

The DoSI dataset can be downloaded via links on [this page](https://structinfer.github.io/download/). Now we provide trajectories that are sufficient to reproduce the results in our benchmarking paper. The rest of DoSI will be made public before the ICLR 2024 conference.

The trajectories should be downloaded, extracted, and saved in ./src/simulations/[type of graphs]/directed or undirected/springs or netsims/. 

## Run Experiments

Each method has different implementation. Please refer to the README in each corresponding subfolder. 

**Caution:** Please be careful with the computational resources required by each method. Some require CPUs, while the other require GPUs.


## Citation

```
@inproceedings{
wang2024benchmarking,
title={Benchmarking Structural Inference Methods for Interacting Dynamical Systems with Synthetic Data},
author={Aoran Wang and Tsz Pan Tong and Andrzej Mizera and Jun Pang},
booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2024},
url={https://openreview.net/forum?id=kKtalvwqBZ}
}
```

## Contact

Aoran Wang: aoran.wang@uni.lu, Tsz Pan Tong: tszpan.tong@uni.lu, Andrzej Mizera: andrzej.mizera@ideas-ncbr.pl, Jun Pang: jun.pang@uni.lu

