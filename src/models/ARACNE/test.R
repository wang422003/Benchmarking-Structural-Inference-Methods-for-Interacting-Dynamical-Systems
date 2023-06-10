library(minet)
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
      exp_data$data = sym_log_normalizer(exp_data$data)
      mim = build.mim(data.frame(t(exp_data$data)), estimator="spearman", disc="equalfreq")
      net = aracne(mim, eps = 0.1)
      write.table(net, file = file.path(parser$save_folder, out_file_name), sep = ",", row.names = FALSE, col.names = FALSE)
    }
  }
}
