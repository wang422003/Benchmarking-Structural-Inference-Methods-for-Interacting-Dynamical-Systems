# Generate Trajectories

Based on the generated underlying interaction graphs stored in ./src/graphs/, we can simulate and sample trajectories with two different types of dynamcial simulations with the scripts in this folder.

To generate Springs and NetSims trajectories without noise, please refer to "generate_trajectories.py".

After the generation of noise-free trajectories, we may refer to "generate_noisy_trajectories.py" to generate trajectories with added Gaussian noise. The noise will be added to the generated noise-free trajectories and save them as NPY files with "\_nX" suffix, where "X" is the level of added Gaussian noise.

"prepare\_EMT_dataset.py" script specifies the pipeline for downloading, preprocessing and constructing trajectory for the EMT dataset.