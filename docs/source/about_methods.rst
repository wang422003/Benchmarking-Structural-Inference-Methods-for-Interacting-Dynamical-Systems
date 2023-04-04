About the Structural Inference Methods
======================================

The structural inference methods benchmarked with StructInf are collected from multiple discinplinaries such as biology and computer science.
We follow the original implementation of these methods, but with slight modification to intergrating data loading and metric calculations.
In the following paragraphs, the implementation of the structural inference methods in this work will be discussed in details.

(TO BE FILLED BY TSZ PAN)
--------------------------

VAE-based Methods
------------------

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
***
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


