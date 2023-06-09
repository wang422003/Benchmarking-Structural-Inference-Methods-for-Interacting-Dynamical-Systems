# PIDC

This repository contains our implementation for the paper *Gene Regulatory Network Inference from Single-Cell Data Using Multivariate Information Measures* accepted by Cell Systems.

The paper online version can be found [here](https://www.cell.com/cell-systems/fulltext/S2405-4712(17)30386-1). 

The official implementation can be found [here](https://github.com/Tchanders/NetworkInference.jl).

## Requirements

To configure the environment, you have to install a Julia executable.

Our environment included:

- ArgParse
- CSV
- DataFrames
- NPZ
- NetworkInference

After installing Julia, you have to install packages in our project by:

1. Install it in Julia interactive session

   ```julia
   julia> using Pkg
   julia> Pkg.instantiate()
   ```

2. Alternatively, install it in Julia REPL mode

   On the shell:

   ```shell
   julia --project=./PIDC/
   ```

   On the Julia REPL mode:

   ```julia
   (PIDC) pkg> instantiate
   ```

## Run experiments

In general, following args are used to select the trajectories to be used for evaluation:

```julia
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
```

Reproduce the results of PIDC in the noise-free trajectories generated by NetSims simulation, and by Brain Networks with 15 nodes, with the first repetition number:

```shell
julia --project=./PIDC/ -- run.jl --save-folder="./results" --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims15r1.npy" &
```

Reproduce the results of PIDC in the noise-free trajectories generated by NetSims simulation, and by Brain Networks with 30 nodes, with the second repetition number:

```shell
julia --project=./PIDC/ -- run.jl --save-folder="./results" --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r2.npy" &
```

Reproduce the results of PIDC in the noisy trajectories generated by NetSims simulation, and by Brain Networks with 50 nodes, with the third repetition number, with two levels of added Gaussian noise:

```shell
julia --project=./PIDC/ -- run.jl --save-folder="./results" --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims50r3_n2.npy" &
```

Reproduce the results of PIDC in the noise-free trajectories generated by NetSims simulation, by Brain Networks with 30 nodes, with the second repetition number, and with 5 time steps:

```shell
julia --project=./PIDC/ -- run.jl --save-folder="./results" --b-network-type="brain_networks" --b-directed --b-simulation-type="netsims" --b-suffix="test_netsims30r2.npy" --b-time-steps=5 &
```

