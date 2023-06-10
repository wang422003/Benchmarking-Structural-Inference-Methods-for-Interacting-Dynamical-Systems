library(ppcor)
source("../helper_functions.R")

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
      model = pcor(data.frame(t(exp_data$data)), method = "spearman")
      model$estimate[1:dim(model$estimate)[1], 1:dim(model$estimate)[1]] = 0
      write.table(model$estimate, file = file.path(parser$save_folder, out_file_name), sep = ",", row.names = FALSE, col.names = FALSE)
    }
  }
}

