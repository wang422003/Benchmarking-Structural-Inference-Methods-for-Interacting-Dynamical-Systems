# RCSI

Code for the submission "Effective and Efficient Structural Inference with Reservoir Computing"

Modification is made for scRNAseq data where no ground truth network provided

## Prerequisites

* [PyTorch](https://pytorch.org/get-started/locally/)
* [scikit-learn](https://scikit-learn.org/stable/getting_started.html) 
* [Scipy](https://scipy.org/install/)
* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [tqdm](https://github.com/tqdm/tqdm)
* [torchsummary](https://pypi.org/project/torch-summary/)
* (Optional) CUDA GPU support. If you don't have it, you can run the training script with "--no-cuda" arg.


## Data generation and Acquisition

### Springs Datasets:

We generate 100\% trajectories with 49 time steps by run:

```
python data/generate_physics_dataset.py --n-balls 10
```

to generates "Springs" by default.

To create the charged particles dataset:

```
python data/generate_physics_dataset.py --simulation 'charged'
```

To create the Kuramoto dataset:

```
python data/generate_kuramoto_dataset.py
```

For synthetic networks and NetSim datasets, we sampled them already and stored in "data/netsims" and "data/Synthetic-H/sampled_data". 


## Example run

For instance to start a session on "Springs" with 100\% trajectories and 49 time steps:

```
train_RC.py --suffix 'springs' --bome --epochs 1000 --RC-portion 1.0 --RC-shuffle --RC-time-steps 49 --timesteps 49
```

For different datasets, use corresponding args to run the experiments, such as on "Springs" with 80\% trajectories and 35 time steps:

```
train_RC.py --suffix 'springs' --bome --epochs 1000 --RC-portion 0.8 --RC-shuffle --RC-time-steps 35 --timesteps 35
```


