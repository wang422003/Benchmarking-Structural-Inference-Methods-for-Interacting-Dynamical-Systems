library(minet)
source("../../helper_functions.R")

parser = get_parser()
parser$save_folder="/home/ttong/bsimds/src/bio_models/ARACNE/results/20230405_001"
parser$b_network_type="chemical_reaction_networks_in_atmosphere"
parser$b-time-steps=49
parser$b_directed=TRUE
parser$b_simulation_type="netsims"
parser$b_suffix="test_netsims15r1.npy"

list_experiment = get_experiment_list_to_be_run(parser)

if (length(list_experiment) == 0){
  stop("There is no experiments to be run. This may because you have specified a wrong data-path or wrong filter name.")
}

out_file_names = get_result_file_names(list_experiment, parser)


for (normalization_method in c("none", "symlog", "unitary", "z-score")){
  exp_data = read_exp_data(list_experiment[1], parser)

  if (normalization_method == "symlog") exp_data$data = sym_log_normalizer(exp_data$data)
  else if (normalization_method == "unitary") exp_data$data = unitary_normalizer(exp_data$data)
  else if (normalization_method == "z-score") exp_data$data = z_score_normalizer(exp_data$data)

  for (min_est in c("mi.empirical", "mi.mm", "mi.shrink", "mi.sg", "pearson","spearman")){
	for (disc in c("equalfreq", "equalwidth", "globalequalwidth")){
		mim = build.mim(data.frame(t(exp_data$data)), estimator=min_est, disc=disc)
#		write.table(mim, file = paste("./mi_",disc,"_",min_est,".csv", sep=""), sep = ",", row.names = FALSE, col.names = FALSE)
			
		for (skip_diag in c(0, 1)){
			net = clr(mim, skipDiagonal = skip_diag)
			write.table(net, file = paste("./inferred_network_",normalization_method,"_",disc,"_",min_est,"_",skip_diag,".csv", sep=""), sep = ",", row.names = FALSE, col.names = FALSE)
#		print("Completed ", "./inferred_network_",disc,"_",min_est,".csv", sep="")
		}
	}
  }
}
