using Pkg, CSV, DataFrames, NPZ, NetworkInference, ArgParse, LinearAlgebra, Statistics
Pkg.instantiate()

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

# ARGS= ["--save-folder", "./results/20230411_001", "--b-network-type", "chemical_reaction_networks_in_atmosphere", "--b-directed", "--b-simulation-type", "netsims", "--b-suffix", "test_netsims15r1.npy"]

parser = parse_args(ARGS, s)

list_experiment = String[]
for (root, dirs, files) in walkdir(parser["data-path"])
    if length(files) != 0 && length(splitpath(root)) - length(splitpath(parser["data-path"])) == 3
        root_list = splitpath(root)
        if (parser["b-network-type"] == "" || parser["b-network-type"] == root_list[end-2]) &&
            ((parser["b-directed"] == true && root_list[end-1] == "directed") || (parser["b-directed"] == false && root_list[end-1] == "undirected")) &&
           (parser["b-simulation-type"] == "" || parser["b-simulation-type"] == root_list[end])
            for file in files
	            if occursin(parser["b-suffix"], file) && occursin(".npy", file) && occursin("edges", file)
		            push!(list_experiment, abspath(joinpath(root, file)))
		        end
	        end
	    end
    end
end

parser["save-folder"] = abspath(parser["save-folder"])
if isdir(parser["save-folder"]) == false && basename(parser["save-folder"]) == ""
    parser["save-folder"] = dirname(parser["save-folder"])
end
if isdir(dirname(parser["save-folder"])) == false
    error("parent folder of save-folder does not exists. Aborting")
end

try
    if isdir(parser["save-folder"]) == false
        mkdir(parser["save-folder"])
    end
catch
end

# normalization functions
function sym_log_normalizer(arr)
    log1p.(abs.(arr)).*sign.(arr)
end

function unitary_normalizer(arr)
    mapslices(x -> x / norm(x), arr, dims=1)
end

function z_score_normalizer(arr)
    (arr .- mean(arr,dims=1)) ./ std(arr,dims=1)
end

feature_dim = 1

for edges_fn in list_experiment
    out_file_name = replace(edges_fn, parser["data-path"] => "")
    for pattern in [r"([/\-\\]+)" => s"_", r"edges_" => s"", r"npy" => s"csv"]
        out_file_name = replace(out_file_name, pattern)
    end
    out_file_name = joinpath(parser["save-folder"], out_file_name)
    if out_file_name in readdir(parser["save-folder"])
        continue
    end

    data_file = [fn for fn in readdir(dirname(edges_fn), join = true) if occursin(replace(splitpath(edges_fn)[end], r"edges_" => s""), fn) && ~occursin("edges", fn)]
    data = []
    n_genes = 0
    for fn in data_file
        new_data = npzread(fn)
        new_data = new_data[:,1:parser["b-time-steps"],feature_dim,:]
        n_genes = size(new_data, 3)
        new_data = reshape(permutedims(new_data, [2,1,3]), :, n_genes)
        new_data = transpose(new_data)

        if size(data)[1] == 0
            data = new_data
        else
            data = [data, new_data]
        end
    end

    # model setting
    discretizer = "uniform_count"; # default value
    estimator = "maximum_likelihood"; # default value
    number_of_bins = round(Int, sqrt(n_genes)) # default value
    base = 2;
    data = sym_log_normalizer(data)

    lines = Array{Any}(undef, size(data, 1), size(data, 2)+1)
    lines[:,1] = 1:n_genes
    lines[:,2:end] = data
    nodes = Array{Node}(undef, n_genes)
    for i in 1:n_genes
        nodes[i] = Node(lines[i:i, 1:end], discretizer, estimator, number_of_bins)
    end
    inferred_network = InferredNetwork(PIDCNetworkInference(), nodes, estimator = estimator, base = base)
    adj_matrix = zeros(n_genes, n_genes);
    for edge in inferred_network.edges
        adj_matrix[parse(Int16, edge.nodes[1].label), parse(Int16, edge.nodes[2].label)] = edge.weight;
    end
    adj_matrix = adj_matrix + transpose(adj_matrix);

    touch(out_file_name)
    open(out_file_name, "w") do io
        CSV.write(io, Tables.table(adj_matrix), writeheader = false);
    end
end
