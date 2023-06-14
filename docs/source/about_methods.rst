**************************************
About the Structural Inference Methods
**************************************

The structural inference methods benchmarked with StructInf are collected from multiple discinplinaries such as biology and computer science.
We follow the original implementation of these methods, but with slight modification to intergrating data loading and metric calculations.
In the following paragraphs, the implementation of the structural inference methods in this work will be discussed in details.

Structural Inference Methods in this Work
=========================================

+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Methods                                                                                                                          | Paper                                                                                                               | Official Implementation                                             | Our Implementation                                                                                                                                           |
+==================================================================================================================================+=====================================================================================================================+=====================================================================+==============================================================================================================================================================+
| ppcor: An R Package for a Fast Calculation to Semi-partial Correlation Coefficients (ppcor)                                      | `Link <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4681537/>`_                                                     | `Link <https://cran.r-project.org/web/packages/ppcor/index.html>`_  | `Link <https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/ppcor>`_               |
+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| TIGRESS: Trustful Inference of Gene REgulation using Stability Selection (TIGRESS)                                               | `Link <https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-6-145>`_                                    | `Link <https://github.com/jpvert/tigress/tree/master>`_             | `Link <https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/TIGRESS>`_             |
+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ARACNE: An Algorithm for the Reconstruction of Gene Regulatory Networks in a Mammalian Cellular Context (ARACNe)                 | `Link <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-S1-S7>`_                            | `Link <https://califano.c2b2.columbia.edu/aracne>`_                 | `Link <https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/ARACNE>`_              |
+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Large-Scale Mapping and Validation of Escherichia coli Transcriptional Regulation from a Compendium of Expression Profiles (CLR) | `Link <https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0050008>`_                             | `Link <https://bioconductor.org/install/>`_                         | `Link <https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/CLR>`_                 |
+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Gene Regulatory Network Inference from Single-Cell Data Using Multivariate Information Measures (PIDC)                           | `Link <https://www.sciencedirect.com/science/article/pii/S2405471217303861>`_                                       | `Link <https://github.com/Tchanders/NetworkInference.jl>`_          | `Link <https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/PIDC>`_                |
+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Inferring Causal Gene Regulatory Networks from Coupled Single-Cell Expression Dynamics Using Scribe (Scribe)                     | `Link <https://www.sciencedirect.com/science/article/pii/S2405471220300363>`_                                       | `Link <https://github.com/aristoteleo/Scribe-py/tree/master>`_      | `Link <https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/scribe>`_              |
+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| dynGENIE3: dynamical GENIE3 for the inference of gene networks from time series expression data (dynGENIE3)                      | `Link <https://www.nature.com/articles/s41598-018-21715-0>`_                                                        | `Link <https://github.com/vahuynh/dynGENIE3/tree/master>`_          | `Link <https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/dynGENIE3>`_           |
+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Inference of gene regulatory networks based on nonlinear ordinary differential equations (XGBGRN)                                | `Link <https://academic.oup.com/bioinformatics/article/36/19/4885/5709036>`_                                        | `Link <https://github.com/lab319/GRNs_nonlinear_ODEs>`_             | `Link <https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/GRNs_nonlinear_ODEs>`_ |
+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Neural Relational Inference for Interacting Systems (NRI)                                                                        | `Link <http://proceedings.mlr.press/v80/kipf18a/kipf18a.pdf>`_                                                      | `Link <http://github.com/ethanfetaya/NRI>`_                         | `Link <https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/NRI>`_                 |
+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Amortized Causal Discovery: Learning to Infer Causal Graphs from Time-Series Data (ACD)                                          | `Link <https://proceedings.mlr.press/v177/lowe22a/lowe22a.pdf>`_                                                    | `Link <https://github.com/loeweX/AmortizedCausalDiscovery>`_        | `Link <https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/ACD>`_                 |
+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Neural Relational Inference with Efficient Message Passing Mechanisms (MPM)                                                      | `Link <https://ojs.aaai.org/index.php/AAAI/article/view/16868>`_                                                    | `Link <https://github.com/hilbert9221/NRI-MPM>`_                    | `Link <https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/MPM>`_                 |
+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Iterative Structural Inference of Directed Graphs (iSIDG)                                                                        | `Link <https://papers.nips.cc/paper_files/paper/2022/file/39717429762da92201a750dd03386920-Paper-Conference.pdf>`_  |                                                                     | `Link <https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/iSIDG>`_               |
+----------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

⋆ Methods based on Classical Statistics
=======================================

ppcor
*****
Following args are used to select the trajectories to be used for evaluation:
::
  parser = add_option(parser, c("--data-path"), type="character", default="/work/projects/bsimds/backup/src/simulations/",
                      help="The folder where data are stored.")
  parser = add_option(parser, c("--save-folder"), type="character", default="",
                      help="The folder where resulting adjacency matrixes are stored.")
  parser = add_option(parser, c("--b-portion"), type="numeric", default=1.0,
                      help="Portion of data to be used in benchmarking.")
  parser = add_option(parser, c("--b-time-steps"), type="integer", default=49L,
                      help="Portion of time series in data to be used in benchmarking")
  parser = add_option(parser, c("--b-network-type"), type="character", default="",
                      help="What is the network type of the graph.")
  parser = add_option(parser, c("--b-directed"), action="store_true", default=FALSE,
                      help="Default choose trajectories from undirected graphs.")
  parser = add_option(parser, c("--b-simulation-type"), type="character", default="",
                      help="Either springs or netsims.")
  parser = add_option(parser, c("--b-suffix"), type="character", default="",
                      help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1. Or "50r1" for 50 nodes, rep 1 and noise free.')

We use the official implementation of ppcor from the R package with a customized wrapper.
Our wrapper will parse multiple arguments to select a set of targeted trajectories for inference, transform trajectories into a suitable format, feed each trajectory into the ppcor algorithm, and store the output into designated directories.
Our implementation can be found at https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/ppcor.
The method is implemented in R with the help of NumPy Python package to store generated trajectories, reticulate from https://github.com/rstudio/reticulate to load Python variables into the R environment, stringr from https://stringr.tidyverse.org for string operation, and optparse from https://github.com/trevorld/r-optparse/tree/master to produce Python-style argument parser.

TIGRESS
*******
Following args are used to select the trajectories to be used for evaluation:
::
  parser = add_option(parser, c("--data-path"), type="character", default="/work/projects/bsimds/backup/src/simulations/",
                      help="The folder where data are stored.")
  parser = add_option(parser, c("--save-folder"), type="character", default="",
                      help="The folder where resulting adjacency matrixes are stored.")
  parser = add_option(parser, c("--b-portion"), type="numeric", default=1.0,
                      help="Portion of data to be used in benchmarking.")
  parser = add_option(parser, c("--b-time-steps"), type="integer", default=49L,
                      help="Portion of time series in data to be used in benchmarking")
  parser = add_option(parser, c("--b-network-type"), type="character", default="",
                      help="What is the network type of the graph.")
  parser = add_option(parser, c("--b-directed"), action="store_true", default=FALSE,
                      help="Default choose trajectories from undirected graphs.")
  parser = add_option(parser, c("--b-simulation-type"), type="character", default="",
                      help="Either springs or netsims.")
  parser = add_option(parser, c("--b-suffix"), type="character", default="",
                      help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1. Or "50r1" for 50 nodes, rep 1 and noise free.')

We use the official implementation of TIGRESS by the author at https://github.com/jpvert/tigress/tree/master with a customized wrapper.
Our wrapper will parse multiple arguments to select a set of targeted trajectories for inference, transform trajectories into a suitable format, feed each trajectory into the TIGRESS algorithm, and store the output into designated directories.
Our implementation can be found at https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/TIGRESS.
The method is implemented in R with the help of NumPy Python package to store generated trajectories, reticulate from https://github.com/rstudio/reticulate to load Python variables into the R environment, stringr for string operation, and optparse from https://github.com/trevorld/r-optparse/tree/master to produce Python-style argument parser.

⋆ Methods based on Information Theory
=====================================

ARACNe
******
Following args are used to select the trajectories to be used for evaluation:
::
  parser = add_option(parser, c("--data-path"), type="character", default="/work/projects/bsimds/backup/src/simulations/",
                      help="The folder where data are stored.")
  parser = add_option(parser, c("--save-folder"), type="character", default="",
                      help="The folder where resulting adjacency matrixes are stored.")
  parser = add_option(parser, c("--b-portion"), type="numeric", default=1.0,
                      help="Portion of data to be used in benchmarking.")
  parser = add_option(parser, c("--b-time-steps"), type="integer", default=49L,
                      help="Portion of time series in data to be used in benchmarking")
  parser = add_option(parser, c("--b-network-type"), type="character", default="",
                      help="What is the network type of the graph.")
  parser = add_option(parser, c("--b-directed"), action="store_true", default=FALSE,
                      help="Default choose trajectories from undirected graphs.")
  parser = add_option(parser, c("--b-simulation-type"), type="character", default="",
                      help="Either springs or netsims.")
  parser = add_option(parser, c("--b-suffix"), type="character", default="",
                      help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1. Or "50r1" for 50 nodes, rep 1 and noise free.')

We use the implementation of ARACNe by the Bioconductor package minet with a customized wrapper.
Our wrapper will parse multiple arguments to select a set of targeted trajectories for inference, transform trajectories into a suitable format, feed each trajectory into the ARACNe algorithm, and store the output into designated directories.
Our implementation can be found at https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/ARACNE.
The method is implemented by minet in R with the help of NumPy Python package to store generated trajectories, reticulate from https://github.com/rstudio/reticulate to load Python variables into the R environment, stringr from https://stringr.tidyverse.org for string operation, and optparse from https://github.com/trevorld/r-optparse/tree/master to produce Python-style argument parser.

CLR
***
Following args are used to select the trajectories to be used for evaluation:
::
  parser = add_option(parser, c("--data-path"), type="character", default="/work/projects/bsimds/backup/src/simulations/",
                      help="The folder where data are stored.")
  parser = add_option(parser, c("--save-folder"), type="character", default="",
                      help="The folder where resulting adjacency matrixes are stored.")
  parser = add_option(parser, c("--b-portion"), type="numeric", default=1.0,
                      help="Portion of data to be used in benchmarking.")
  parser = add_option(parser, c("--b-time-steps"), type="integer", default=49L,
                      help="Portion of time series in data to be used in benchmarking")
  parser = add_option(parser, c("--b-network-type"), type="character", default="",
                      help="What is the network type of the graph.")
  parser = add_option(parser, c("--b-directed"), action="store_true", default=FALSE,
                      help="Default choose trajectories from undirected graphs.")
  parser = add_option(parser, c("--b-simulation-type"), type="character", default="",
                      help="Either springs or netsims.")
  parser = add_option(parser, c("--b-suffix"), type="character", default="",
                      help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1. Or "50r1" for 50 nodes, rep 1 and noise free.')

We use the implementation of CLR by the Bioconductor package minet with a customized wrapper.
Our wrapper will parse multiple arguments to select a set of targeted trajectories for inference, transform trajectories into a suitable format, feed each trajectory into the CLR algorithm, and store the output into designated directories.
Our implementation can be found at https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/CLR.
The method is implemented by minet in R with the help of NumPy Python package to store generated trajectories, reticulate from https://github.com/rstudio/reticulate to load Python variables into the R environment, stringr from https://stringr.tidyverse.org for string operation, and optparse from https://github.com/trevorld/r-optparse/tree/master to produce Python-style argument parser.

PIDC
****
Following args are used to select the trajectories to be used for evaluation:
::
  s = ArgParseSettings()
  @add_arg_table s begin
      "--data-path"
          help = "The folder where data are stored."
        arg_type = String
        default = "/work/projects/bsimds/backup/src/simulations/"
      "--save-folder"
          help = "The folder where resulting adjacency matrixes are stored."
          arg_type = String
          required = true
      "--b-portion"
        help = "Portion of data to be used in benchmarking."
        arg_type = Float64
        default = 1.0
      "--b-time-steps"
        help = "Portion of data to be used in benchmarking."
        arg_type = Int
        default = 49
      "--b-shuffle"
        help = "Shuffle the data for benchmarking?"
        action = :store_true
        default = false
      "--b-network-type"
          help = "What is the network type of the graph."
          arg_type = String
        default = ""
      "--b-directed"
        help = "Default choose trajectories from undirected graphs."
        action = :store_true
      "--b-simulation-type"
        help = "Either springs or netsims."
        arg_type = String
        default = ""
      "--b-suffix"
          help = "The rest to locate the exact trajectories. E.g. \"50r1_n1\" for 50 nodes, rep 1 and noise level 1. Or \"50r1\" for 50 nodes, rep 1 and noise free."
          arg_type = String
        default = ""
  end

We use the official implementation of PIDC by the author at https://github.com/Tchanders/NetworkInference.jl with a customized wrapper.
Our wrapper will parse multiple arguments to select a set of targeted trajectories for inference, transform trajectories into a suitable format, feed each trajectory into the PIDC algorithm, and store the output into designated directories.
Our implementation can be found at https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/PIDC.
The method is implemented in Julia with the help of NumPy Python package to store generated trajectories, NPZ.jl from https://github.com/fhs/NPZ.jl to load .npy into the Julia environment, stringr from https://stringr.tidyverse.org for string operation, and optparse from https://github.com/trevorld/r-optparse/tree/master to produce Python-style argument parser.

Scribe
******
Following args are used to select the trajectories to be used for evaluation:
::
  parser.add_argument('--data-path', type=str,
                      default="/work/projects/bsimds/backup/src/simulations/",
                      help="The folder where data are stored.")
  parser.add_argument('--save-folder', type=str, required=True,
                      help="The folder where resulting adjacency matrixes are stored.")
  parser.add_argument('--b-portion', type=float, default=1.0,
                      help='Portion of data to be used in benchmarking.')
  parser.add_argument('--b-time-steps', type=int, default=49,
                      help='Portion of time series in data to be used in benchmarking.')
  parser.add_argument('--b-shuffle', action='store_true', default=False,
                      help='Shuffle the data for benchmarking?')
  parser.add_argument('--b-network-type', type=str, default='',
                      help='What is the network type of the graph.')
  parser.add_argument('--b-directed', action='store_true', default=False,
                      help='Default choose trajectories from undirected graphs.')
  parser.add_argument('--b-simulation-type', type=str, default='',
                      help='Either springs or netsims.')
  parser.add_argument('--b-suffix', type=str, default='',
                  help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1. Or "50r1" for 50 nodes, rep 1 and noise free.')
  parser.add_argument('--pct-cpu', type=float, default=1.0,
                      help='Percentage of number of CPUs to be used.')

We optimize the official implementation of scribe by the author at https://github.com/aristoteleo/Scribe-py/tree/master with a customized wrapper.
Our wrapper will parse multiple arguments to select a set of targeted trajectories for inference, transform trajectories into a suitable format, feed each trajectory into the scribe algorithm, and store the output into designated directories.
Our implementation has customized causal_network.py and information_estimators.py scripts so as to modify the hyperparameters directly from command line arguments.
We also have optimized the parallel support and computation efficiency and kept minimal functionality for benchmarking purposes, at the same time maintaining its general mechanism.
Our implementation can be found at https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/scribe.
The method is implemented in Python with the help of NumPy package to store generated trajectories and tqdm from https://github.com/tqdm/tqdm to create progress bars.

⋆ Methods based on Tree Algorithms
==================================

dynGENIE3
*********
Following args are used to select the trajectories to be used for evaluation:
::
  parser.add_argument('--data-path', type=str,
                      default="/work/projects/bsimds/backup/src/simulations/",
                      help="The folder where data are stored.")
  parser.add_argument('--save-folder', type=str, required=True,
                      help="The folder where resulting adjacency matrixes are stored.")
  parser.add_argument('--b-portion', type=float, default=1.0,
                      help='Portion of data to be used in benchmarking.')
  parser.add_argument('--b-time-steps', type=int, default=49,
                      help='Portion of time series in data to be used in benchmarking.')
  parser.add_argument('--b-shuffle', action='store_true', default=False,
                      help='Shuffle the data for benchmarking?')
  parser.add_argument('--b-network-type', type=str, default='',
                      help='What is the network type of the graph.')
  parser.add_argument('--b-directed', action='store_true', default=False,
                      help='Default choose trajectories from undirected graphs.')
  parser.add_argument('--b-simulation-type', type=str, default='',
                      help='Either springs or netsims.')
  parser.add_argument('--b-suffix', type=str, default='',
                  help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1. Or "50r1" for 50 nodes, rep 1 and noise free.')
  parser.add_argument('--pct-cpu', type=float, default=1.0,
                      help='Percentage of number of CPUs to be used.')

We optimize the official Python implementation of dynGENIE3 by the author at https://github.com/vahuynh/dynGENIE3/tree/master with a customized wrapper.
Our wrapper will parse multiple arguments to select a set of targeted trajectories for inference, transform trajectories into a suitable format, feed each trajectory into the dynGENIE3 algorithm, and store the output into designated directories.
Following the principle of maintaining dynGENIE's general mechanism, we have modified the dynGENIE3.py script so as to tune the hyperparameters directly from command line arguments, increase computation efficiency on big datasets, enable calculation of self-influence, and retain minimal functionality for benchmarking purposes.
Our implementation can be found at https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/dynGENIE3.
The method is implemented in Python with the help of NumPy package to store generated trajectories.

XGBGRN
******
Following args are used to select the trajectories to be used for evaluation:
::
  parser.add_argument('--data-path', type=str,
                      default="/work/projects/bsimds/backup/src/simulations/",
                      help="The folder where data are stored.")
  parser.add_argument('--save-folder', type=str, required=True,
                      help="The folder where resulting adjacency matrixes are stored.")
  parser.add_argument('--b-portion', type=float, default=1.0,
                      help='Portion of data to be used in benchmarking.')
  parser.add_argument('--b-time-steps', type=int, default=49,
                      help='Portion of time series in data to be used in benchmarking.')
  parser.add_argument('--b-shuffle', action='store_true', default=False,
                      help='Shuffle the data for benchmarking?')
  parser.add_argument('--b-network-type', type=str, default='',
                      help='What is the network type of the graph.')
  parser.add_argument('--b-directed', action='store_true', default=False,
                      help='Default choose trajectories from undirected graphs.')
  parser.add_argument('--b-simulation-type', type=str, default='',
                      help='Either springs or netsims.')
  parser.add_argument('--b-suffix', type=str, default='',
                  help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1. Or "50r1" for 50 nodes, rep 1 and noise free.')
  parser.add_argument('--pct-cpu', type=float, default=1.0,
                      help='Percentage of number of CPUs to be used.')

We use the official implementation of XGBGRN by the author at https://github.com/lab319/GRNs_nonlinear_ODEs with a customized wrapper.
Our wrapper will parse multiple arguments to select a set of targeted trajectories for inference, transform trajectories into a suitable format, feed each trajectory into the XGBGRN algorithm, and store the output into designated directories.
Our implementation can be found at https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/GRNs_nonlinear_ODEs.
The method is implemented in Python with the help of NumPy package to store generated trajectories.

⋆ Methods based on VAEs
=======================

In general, we added following arguments to the argparse variable in these methods:
::
  parser.add_argument('--save-probs', action='store_true', default=False,
                      help='Save the probs during test.')
  parser.add_argument('--b-portion', type=float, default=1.0,
                      help='Portion of data to be used in benchmarking.')
  parser.add_argument('--b-time-steps', type=int, default=49,
                      help='Portion of time series in data to be used in benchmarking.')
  parser.add_argument('--b-shuffle', action='store_true', default=False,
                      help='Shuffle the data for benchmarking.')
  parser.add_argument('--data-path', type=str, default='',
                      help='Where to load the data. May input the paths to edges_train of the data.')
  parser.add_argument('--b-network-type', type=str, default='',
                      help='What is the network type of the graph.')
  parser.add_argument('--b-directed', action='store_true', default=False,
                      help='Default choose trajectories from undirected graphs.')
  parser.add_argument('--b-simulation-type', type=str, default='',
                      help='Either springs or netsims.')
  parser.add_argument('--b-suffix', type=str, default='',
      help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1.'
           ' Or "50r1" for 50 nodes, rep 1 and noise free.')


NRI
****
We use the official implementation code by the author from https://github.com/ethanfetaya/NRI with customized data loaders for our chosen datasets.
The customized data loaders are named "load\_customized\_springs\_data" and "load\_customized\_netsims\_data". Both of them are implemented in the "utils.py" file.
The metric calculation pipeline is integrated into the "test" function.
Besides that, the remaining part are in consistent with its official implementation.
The code of our implementation can be found at https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/NRI .

ACD
***
We use the official implementation code by the author https://github.com/loeweX/AmortizedCausalDiscovery with a customized data loader for our datasets. 
The customized data loader is named "load\_data\_customized", and is implemented in "data\_loader.py".
The metric calculation pipeline is integrated into the function "forward\_pass\_and\_eval" of "foward\_pass\_and\_eval.py" file.
Besides that, the remaining part are in consistent with its official implementation.
The code of our implementation can be found at https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/ACD .

MPM
***
We use the official implementation code by the author at https://github.com/hilbert9221/NRI-MPM with a customized data loader for our chosen datasets.
The customized data loader function is named "load\_customized\_data", and with data preprocessing functions "load\_nri" and "load\_netsims".
The first function is implemented in "run.py", while the rest are implemented in "load.py".
The metric calculation pipelines are integrated into the "test" function of "XNRIIns" class in "XNRI.py" file.
Besides that, the remaining part are in consistent with its official implementation.
The code of our implementation can be found at https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/MPM .

iSIDG
******
We use the official implementation sent by the authors.
We modified it with a customized data loader function: "load\_data\_benchmark", which is implemented in "utils.py".
Besides that, the remaining part are in consistent with its official implementation.
The code of our implementation can be found at https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/tree/main/src/models/iSIDG .


