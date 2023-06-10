source("../../helper_functions.R")
source("../stabilityselection.R")
source("../tigress.R")
library(lars)
library(optparse)

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
parser = add_option(parser, c("--normalization-method"), type="character", default="none")
parser = add_option(parser, c("--alpha"), type="numeric", default=0.2)
parser = add_option(parser, c("--nstepsLARS"), type="numeric", default=5)
parser = add_option(parser, c("--nsplit"), type="numeric", default=100)
parser = add_option(parser, c("--normalizeexp"), action="store_true", default=FALSE)
parser = add_option(parser, c("--scoring"), type="character", default="area")
parser = parse_args(parser, convert_hyphens_to_underscores  = TRUE)

list_experiment = get_experiment_list_to_be_run(parser)

if (length(list_experiment) == 0){
  stop("There is no experiments to be run. This may because you have specified a wrong data-path or wrong filter name.")
}

out_file_names = get_result_file_names(list_experiment, parser)

exp_data = read_exp_data(list_experiment[1], parser)
rownames(exp_data$data) = 1:dim(exp_data$data)[1]

if (parser$normalization_method == "symlog") exp_data$data = sym_log_normalizer(exp_data$data)
if (parser$normalization_method == "unitary") exp_data$data = unitary_normalizer(exp_data$data)
if (parser$normalization_method == "z-score") exp_data$data = z_score_normalizer(exp_data$data)
  
net = tigress(t(exp_data$data), alpha=parser$alpha, nstepsLARS=parser$nstepsLARS, nsplit=parser$nsplit, normalizeexp=parser$normalizeexp, scoring=parser$scoring, allsteps=FALSE, usemulticore=TRUE)
write.table(net, file = paste("./inferred_network_",parser$normalization_method,"_",parser$alpha,"_",parser$nstepsLARS,"_",parser$nsplit,"_",parser$normalizeexp,"_",parser$scoring,".csv", sep=""), sep = ",", row.names = FALSE, col.names = FALSE)