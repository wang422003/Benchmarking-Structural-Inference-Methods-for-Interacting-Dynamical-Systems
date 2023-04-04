How to Reproduce the Results
============================

Before the reproduction of the results in out benchmark, please follow the instructions in :doc:`/about_dataset` to download the datasets, and store them correctly under every subfolder.
Then the structural inference methods in the benchmark and their results can be reproduced with following steps.


(To Be Filled By Tsz Pan)
--------------------------






VAE-based Methods
------------------

NRI
****
Please install the required packages first.

**Requirements**

- Python >= 3.8
- Numpy >= 1.23.4
- pandas >= 1.5.1
- matplotlib >= 3.6.2
- sklearn >= 0.0.post1
- torch >= 1.13.1
- torchinfo >= 1.7.2
- tqdm >= 4.64.1

**Arguments**

- b-network-type: name of the graph type (in full name)
- b-directed: if called, will load data from directed graphs
- b-simulation-type: springs or netsims
- b-suffix: choose graph with node X, the Y repetition and with noise level K with format "XrY_nK". If use noise-free, omit "_nK"


**Reproduction Examples**

Run NRI with “chemical reaction networks in atmosphere (CRNA)”, “directed”, “15 nodes”, “springs simulation”, “noise-free”, and “the first repetition” :
::
  $> cd /src/models/NRI/
  $> python3 train.py --b-network-type 'chemical_reaction_networks_in_atmosphere' --b-directed --b-simulation-type 'springs' --b-suffix '15r1'

Run NRI with “brain networks (BN)”, “directed”, “netsims simulation”, “30 nodes”, “noise-free”, and “the second repetition”:
::
  $> cd /src/models/NRI/
  $> python3 train.py --b-network-type 'brain_networks' --b-directed --b-simulation-type 'netsims' --b-suffix '30r2'
  
Run NRI with “landscape networks (LN)”, “directed”, “netsims simulation”, “50 nodes”, “the third repetition”, and “noise level 2”:
::
  $> cd /src/models/NRI/
  $> python3 train.py --b-network-type 'landscape_networks' --b-simulation-type 'netsims' --b-suffix '50r3_n2'
  

ACD
****
Please install the required packages first.

**Requirements**

- Python >= 3.8
- Numpy >= 1.23.4
- pandas >= 1.5.1
- scipy >= 1.9.3
- sklearn >= 0.0.post1
- torch >= 1.13.1
- torchinfo >= 1.7.2
- tqdm >= 4.64.1

**Arguments**

- b-network-type: name of the graph type (in full name)
- b-directed: if called, will load data from directed graphs
- b-simulation-type: springs or netsims
- b-suffix: choose graph with node X, the Y repetition and with noise level K with format "XrY_nK". If use noise-free, omit "_nK"


**Reproduction Examples**

Run ACD with “chemical reaction networks in atmosphere (CRNA)”, “directed”, “15 nodes”, “springs simulation”, “noise-free”, and “the first repetition” :
::
  $> cd /src/models/ACD/
  $> python3 train.py --b-network-type 'chemical_reaction_networks_in_atmosphere' --b-directed --b-simulation-type 'springs' --b-suffix '15r1'

Run ACD with “brain networks (BN)”, “directed”, “netsims simulation”, “30 nodes”, “noise-free”, and “the second repetition”:
::
  $> cd /src/models/ACD/
  $> python3 train.py --b-network-type 'brain_networks' --b-directed --b-simulation-type 'netsims' --b-suffix '30r2'
  
Run ACD with “landscape networks (LN)”, “directed”, “netsims simulation”, “50 nodes”, “the third repetition”, and “noise level 2”:
::
  $> cd /src/models/ACD/
  $> python3 train.py --b-network-type 'landscape_networks' --b-simulation-type 'netsims' --b-suffix '50r3_n2'
  
MPM
****
Please install the required packages first.

**Requirements**

- Python >= 3.8
- Numpy >= 1.23.4
- scipy >= 1.9.3
- sklearn >= 0.0.post1
- torch >= 1.13.1
- torch-geometric >= 2.2.0
- torchinfo >= 1.7.2
- tqdm >= 4.64.1

**Arguments**

- b-network-type: name of the graph type (in full name)
- b-directed: if called, will load data from directed graphs
- b-simulation-type: springs or netsims
- b-suffix: choose graph with node X, the Y repetition and with noise level K with format "XrY_nK". If use noise-free, omit "_nK"


**Reproduction Examples**

Run ACD with “chemical reaction networks in atmosphere (CRNA)”, “directed”, “15 nodes”, “springs simulation”, “noise-free”, and “the first repetition” :
::
  $> cd /src/models/MPM/
  $> python3 run.py --b-network-type 'chemical_reaction_networks_in_atmosphere' --b-directed --b-simulation-type 'springs' --b-suffix '15r1'

Run ACD with “brain networks (BN)”, “directed”, “netsims simulation”, “30 nodes”, “noise-free”, and “the second repetition”:
::
  $> cd /src/models/MPM/
  $> python3 run.py --b-network-type 'brain_networks' --b-directed --b-simulation-type 'netsims' --b-suffix '30r2'
  
Run ACD with “landscape networks (LN)”, “directed”, “netsims simulation”, “50 nodes”, “the third repetition”, and “noise level 2”:
::
  $> cd /src/models/ACD/
  $> python3 run.py --b-network-type 'landscape_networks' --b-simulation-type 'netsims' --b-suffix '50r3_n2'
  
iSIDG
******
Please install the required packages first.

**Requirements**

- Python >= 3.8
- Numpy >= 1.23.4
- pandas >= 1.5.1
- matplotlib >= 3.6.2
- sklearn >= 0.0.post1
- torch >= 1.13.1
- torchinfo >= 1.7.2
- tqdm >= 4.64.1

**Arguments**

- b-network-type: name of the graph type (in full name)
- b-directed: if called, will load data from directed graphs
- b-simulation-type: springs or netsims
- b-suffix: choose graph with node X, the Y repetition and with noise level K with format "XrY_nK". If use noise-free, omit "_nK"


**Reproduction Examples**

Run NRI with “chemical reaction networks in atmosphere (CRNA)”, “directed”, “15 nodes”, “springs simulation”, “noise-free”, and “the first repetition” :
::
  $> cd /src/models/iSIDG/
  $> python3 train.py --b-network-type 'chemical_reaction_networks_in_atmosphere' --b-directed --b-simulation-type 'springs' --b-suffix '15r1'

Run NRI with “brain networks (BN)”, “directed”, “netsims simulation”, “30 nodes”, “noise-free”, and “the second repetition”:
::
  $> cd /src/models/iSIDG/
  $> python3 train.py --b-network-type 'brain_networks' --b-directed --b-simulation-type 'netsims' --b-suffix '30r2'
  
Run NRI with “landscape networks (LN)”, “directed”, “netsims simulation”, “50 nodes”, “the third repetition”, and “noise level 2”:
::
  $> cd /src/models/iSIDG/
  $> python3 train.py --b-network-type 'landscape_networks' --b-simulation-type 'netsims' --b-suffix '50r3_n2'

