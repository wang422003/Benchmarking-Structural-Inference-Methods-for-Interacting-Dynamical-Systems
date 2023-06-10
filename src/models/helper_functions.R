#install.packages("stringi")
#install.packages("https://cran.r-project.org/src/contrib/stringi_1.7.12.tar.gz")
library(reticulate)
library(stringr)
library(optparse)

reticulate::use_condaenv("dynGENIE3")
np = import("numpy")
feature_dim = 1

read_exp_data = function(edges_file_path, parser){
  current_folder = strsplit(edges_file_path, .Platform$file.sep)[[1]]
  if (length(current_folder) != 1){
    current_folder = paste(current_folder[1:length(current_folder)-1], collapse = .Platform$file.sep)
  } else {
    current_folder = "."
  }

  edge_file_name = tail(strsplit(edges_file_path, .Platform$file.sep)[[1]], 1)
  data_file = list.files(current_folder, pattern = substr(edge_file_name, 6, stop = 1000000L), full.names = TRUE)
  data_file = data_file[-grep("edges", data_file)]
  
  # all .npy files have shape (n_trajectory, n_timestep, n_feature, n_node)
  out = list(data = NULL, ground_truth = NULL, current_folder=current_folder, edge_file_name=edge_file_name)
  for (file in data_file){
    data = np$load(file)
    data = data[,1:parser$b_time_steps,feature_dim,]
    data = np$reshape(data, c(-1L, dim(data)[3]))
    data = t(data)
    if (is.null(out$data)){
      out$data = data
    } else{
      out$data = rbind(out$data, data)
    }
  }
  
  out$ground_truth = np$load(edges_file_path)
  
  return(out)
}

get_parser = function(){
  parser = OptionParser()
  
  parser = add_option(parser, c("--data-path"), type="character", default="/work/projects/bsimds/backup/src/simulations/",
                      help="The folder where data are stored.")
  parser = add_option(parser, c("--save-folder"), type="character", default="",
                      help="The folder where resulting adjacency matrixes are stored.")
  parser = add_option(parser, c("--b-portion"), type="numeric", default=1.0,
                      help="Portion of data to be used in benchmarking.")
  parser = add_option(parser, c("--b-time-steps"), type="integer", default=49L,
                      help="Portion of time series in data to be used in benchmarking")
  parser = add_option(parser, c("--b-network-type"), type="character", default="",
                      help="What is the network type of the graph.")
  parser = add_option(parser, c("--b-directed"), action="store_true", default=FALSE,
                      help="Default choose trajectories from undirected graphs.")
  parser = add_option(parser, c("--b-simulation-type"), type="character", default="",
                      help="Either springs or netsims.")
  parser = add_option(parser, c("--b-suffix"), type="character", default="",
                      help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1. Or "50r1" for 50 nodes, rep 1 and noise free.')
  parser = add_option(parser, c("--pct-cpu"), type="numeric", default=1.0,
  		      help='Percentage of number of CPUs to be used.')
  return(parse_args(parser, convert_hyphens_to_underscores  = TRUE))
}

get_experiment_list_to_be_run = function(parser){
  if (parser$save_folder == ""){
    stop("No save_folder provided")
  }
  dir.create(parser$save_folder, showWarnings = FALSE)
  
  list_file = list.files(parser$data_path, recursive = TRUE, pattern = ".npy", full.names = TRUE)
  list_file = list_file[str_count(list_file, .Platform$file.sep) == str_count(parser$data_path, .Platform$file.sep) + 4]
  list_ground_truth = list_file[grep("edges", list_file)]
  
  if (parser$b_network_type != ""){
    #pattern = paste(.Platform$file.sep, parser$b_network_type, .Platform$file.sep, sep = "")
    pattern = parser$b_network_type
    list_ground_truth = list_ground_truth[grep(pattern=pattern, x=list_ground_truth)]
  }
  
  if (parser$b_directed == FALSE){
    list_ground_truth = list_ground_truth[grep(pattern="/undirected/", x=list_ground_truth)]
  }
  else{
    list_ground_truth = list_ground_truth[grep(pattern="/directed/", x=list_ground_truth)]
  }

  if (parser$b_simulation_type != ""){
    #pattern = cat(.Platform$file.sep, parser$b_simulation_type, .Platform$file.sep, sep="")
    pattern = parser$b_simulation_type
    list_ground_truth = list_ground_truth[grep(pattern=pattern, x=list_ground_truth)]
  }
  
  if (parser$b_suffix != ""){
    list_ground_truth = list_ground_truth[grep(pattern=parser$b_suffix, x=list_ground_truth)]
  }
  
  return(list_ground_truth)
}

get_result_file_names = function(list_experiment, parser){
  out_file_names = str_sub(list_experiment, start = nchar(parser$data_path)+2)
  out_file_names = str_replace_all(out_file_names, "npy", "csv")
  out_file_names = str_replace_all(out_file_names, "[/'\\-\\\\]", "_")
  out_file_names = str_replace_all(out_file_names, "edges_", "")
  return(out_file_names)
}

sym_log_normalizer = function(arr){
  return(log(1+abs(arr)) * sign(arr))
}

unitary_normalizer = function(arr){
  return(arr / apply(arr, 2, function(x) sqrt(sum(x^2))))
}

z_score_normalizer = function(arr){
  return(apply(exp_data$data, 2, function(x) (x-mean(x))/sd(x)))
}