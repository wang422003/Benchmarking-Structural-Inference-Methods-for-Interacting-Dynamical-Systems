# :sparkles: Benchmarking Structural Inference Methods for Interacting Dynamical Systems with Synthetic Data :sparkles:


[![Documentation Status](https://readthedocs.org/projects/benchmarking-structural-inference-methods/badge/?version=latest)](https://benchmarking-structural-inference-methods.readthedocs.io/en/latest/?badge=latest)
![Last Commit](https://img.shields.io/github/last-commit/divelab/DIG)
[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by]. 

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey

[**Documentation**](https://benchmarking-structural-inference-methods.readthedocs.io/) | [**Project Website**](https://structinfer.github.io/)

This repo maintains and updates benchmark on structural inference methods for interacting dynamical systems with synthetic data which is submitted to NeurIPS 2023 Datasets and Benchmarks Track. :smile:

## Installation

Clone the reporitory:
```
$> git clone https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems.git 
```

Different methods require different programming languages and different packages. Please refer to the README in each sub-folder (our implementation) and install the requirements:

| Methods                                                      | Paper                                                        | Official Implementation                                      | Our Implementation                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ppcor: An R Package for a Fast Calculation to Semi-partial Correlation Coefficients (ppcor) | [Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4681537/) | [Link](https://cran.r-project.org/web/packages/ppcor/index.html) | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/ppcor) |
| TIGRESS: Trustful Inference of Gene REgulation using Stability Selection (TIGRESS) | [Link](https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-6-145) | [Link](https://github.com/jpvert/tigress/tree/master)        | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/TIGRESS) |
| ARACNE: An Algorithm for the Reconstruction of Gene Regulatory Networks in a Mammalian Cellular Context (ARACNe) | [Link](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-S1-S7) | [Link](https://califano.c2b2.columbia.edu/aracne)            | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/ARACNE) |
| Large-Scale Mapping and Validation of Escherichia coli Transcriptional Regulation from a Compendium of Expression Profiles (CLR) | [Link](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0050008) | [Link](https://bioconductor.org/install/)                    | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/CLR) |
| Gene Regulatory Network Inference from Single-Cell Data Using Multivariate Information Measures (PIDC) | [Link](https://www.sciencedirect.com/science/article/pii/S2405471217303861) | [Link](https://github.com/Tchanders/NetworkInference.jl)     | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/PIDC) |
| Inferring Causal Gene Regulatory Networks from Coupled Single-Cell Expression Dynamics Using Scribe (Scribe) | [Link](https://www.sciencedirect.com/science/article/pii/S2405471220300363) | [Link](https://github.com/aristoteleo/Scribe-py/tree/master) | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/scribe) |
| dynGENIE3: dynamical GENIE3 for the inference of gene networks from time series expression data (dynGENIE3) | [Link](https://www.nature.com/articles/s41598-018-21715-0)   | [Link](https://github.com/vahuynh/dynGENIE3/tree/master)     | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/dynGENIE3) |
| Inference of gene regulatory networks based on nonlinear ordinary differential equations (XGBGRN) | [Link](https://academic.oup.com/bioinformatics/article/36/19/4885/5709036) | [Link](https://github.com/lab319/GRNs_nonlinear_ODEs)        | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/GRNs_nonlinear_ODEs) |
| Neural Relational Inference for Interacting Systems (NRI)    | [Link](http://proceedings.mlr.press/v80/kipf18a/kipf18a.pdf) | [Link](http://github.com/ethanfetaya/NRI)                    | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/NRI) |
| Amortized Causal Discovery: Learning to Infer Causal Graphs from Time-Series Data (ACD) | [Link](https://proceedings.mlr.press/v177/lowe22a/lowe22a.pdf) | [Link](https://github.com/loeweX/AmortizedCausalDiscovery)   | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/ACD) |
| Neural Relational Inference with Efficient Message Passing Mechanisms (MPM) | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/16868) | [Link](https://github.com/hilbert9221/NRI-MPM)               | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/MPM) |
| Iterative Structural Inference of Directed Graphs (iSIDG)    | [Link](https://papers.nips.cc/paper_files/paper/2022/file/39717429762da92201a750dd03386920-Paper-Conference.pdf) |                                                              | [Link](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/iSIDG) |

## Dataset

The DoSI dataset can be downloaded via links on [this page](https://structinfer.github.io/download/). Now we provide trajectories that are sufficient to reproduce the results in our benchmarking paper. The rest of DoSI will be made public before the NeurIPS 2023 conference.

The trajectories should be downloaded, extracted, and saved in ./src/simulations/[type of graphs]/directed or undirected/springs or netsims/. A detailed instruction can be found at [this page](https://benchmarking-structural-inference-methods.readthedocs.io/en/latest/about_dataset.html#downloading-npy-datasets).

## Run Experiments

Each method has different implementation. Please refer to the README in each corresponding subfolder. 

**Caution:** Please be careful with the computational resources required by each method. Some require CPUs, while the other require GPUs.


## Citation

TBA

## Contact

Please feel free to contact [Aoran Wang](mailto:aoran.wang@uni.lu), [Tsz Pan Tong](mailto:tszpan.tong@uni.lu), or [Jun Pang](mailto:jun.pang@uni.lu)!

