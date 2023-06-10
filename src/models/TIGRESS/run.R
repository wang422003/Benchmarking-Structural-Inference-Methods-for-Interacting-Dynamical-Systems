source("../helper_functions.R")
source("stabilityselection.R")
source("tigress.R")
library(lars)

parser = get_parser()

list_experiment = get_experiment_list_to_be_run(parser)

if (length(list_experiment) == 0){
  stop("There is no experiments to be run. This may because you have specified a wrong data-path or wrong filter name.")
}

out_file_names = get_result_file_names(list_experiment, parser)

if (length(out_file_names) != 0){
  for (experiment_id in 1:length(list_experiment)){
    out_file_name = out_file_names[experiment_id]
    print(paste("Processing ", out_file_name)) 

    if (out_file_name %in% list.files(parser$save_folder, full.names = FALSE) == FALSE){
      exp_data = read_exp_data(list_experiment[experiment_id], parser)
      rownames(exp_data$data) = 1:dim(exp_data$data)[1]
      exp_data$data = sym_log_normalizer(exp_data$data)
      net = tigress(t(exp_data$data), alpha=0.5, nstepsLARS=5, nsplit=500, normalizeexp=FALSE, scoring="area", allsteps=FALSE, usemulticore=TRUE)
      write.table(net, file = file.path(parser$save_folder, out_file_name), sep = ",", row.names = FALSE, col.names = FALSE)
    }
  }
}