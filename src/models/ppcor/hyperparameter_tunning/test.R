library(ppcor)
source("../../helper_functions.R")

parser = get_parser()
parser$save_folder="/home/ttong/bsimds/src/bio_models/ARACNE/results/20230405_001"
parser$b_time_steps=49
parser$b_network_type="chemical_reaction_networks_in_atmosphere"
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
  
  for (method in c("pearson", "spearman")){
    for (func in c("spcor", "pcor")){
    
      if (func == "spcor") model = spcor(data.frame(t(exp_data$data)), method = method)
      else model = pcor(data.frame(t(exp_data$data)), method = method)
  		
      write.table(model$estimate, file = paste("./inferred_network_",normalization_method,"_",method,"_",func,".csv", sep=""), sep = ",", row.names = FALSE, col.names = FALSE)
      }
  }
}
