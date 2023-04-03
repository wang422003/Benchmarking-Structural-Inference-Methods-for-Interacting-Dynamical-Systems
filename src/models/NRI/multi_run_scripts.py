import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--b-network-type', type=str, default='chemical_reaction_networks_in_atmosphere',
                    help='What is the network type of the graph.')
parser.add_argument('--b-num-nodes', type=int, default=15,
                    help='The rest to locate the exact trajectories. E.g. "50" for 50 nodes.'
                         ' Or "30" for 30 nodes.')
parser.add_argument('--b-simulation-type', type=str, default='netsims',
                    help='Either springs or netsims.')

args = parser.parse_args()

if args.b_network_type == 'chemical_reaction_networks_in_atmosphere':
    str_network = 'CRN'
elif args.b_network_type == 'gene_coexpression_networks':
    str_network = 'GCN'
elif args.b_network_type == 'gene_regulatory_networks':
    str_network = 'GRN'
elif args.b_network_type == 'intercellular_networks':
    str_network = 'IN'
elif args.b_network_type == 'landscape_networks':
    str_network = 'LN'
elif args.b_network_type == 'brain_networks':
    str_network = 'BN'
elif args.b_network_type == 'food_webs':
    str_network = 'FW'
elif args.b_network_type == 'man-made_organic_reaction_networks':
    str_network = 'MMORN'
elif args.b_network_type == 'reaction_networks_inside_living_organism':
    str_network = 'RNILO'
elif args.b_network_type == 'social_networks':
    str_network = 'SN'
elif args.b_network_type == 'vascular_networks':
    str_network = 'VN'
else:
    str_network = ''

if args.b_simulation_type == 'springs':
    str_sim = 'SP'
elif args.b_simulation_type == 'netsims':
    str_sim = 'NS'
else:
    str_sim = ''

currt_path = os.getcwd()

# bash_list = [
#     'train_NRI_CRN_30_D_NS_R1.sh',
#     'train_NRI_CRN_30_D_NS_R2.sh',
#     'train_NRI_CRN_30_D_NS_R3.sh',
#     'train_NRI_CRN_30_D_NS_R1_seed10.sh',
#     'train_NRI_CRN_30_D_NS_R2_seed10.sh',
#     'train_NRI_CRN_30_D_NS_R3_seed10.sh',
#     'train_NRI_CRN_30_D_NS_R1_seed144.sh',
#     'train_NRI_CRN_30_D_NS_R2_seed144.sh',
#     'train_NRI_CRN_30_D_NS_R3_seed144.sh',
# ]

bash_list = [
    '_R1.sh',
    '_R2.sh',
    '_R3.sh',
    '_R1_seed10.sh',
    '_R2_seed10.sh',
    '_R3_seed10.sh',
    '_R1_seed144.sh',
    '_R2_seed144.sh',
    '_R3_seed144.sh',
]

for i, name in enumerate(bash_list):
    bash_list[i] = 'train_NRI_' + str_network + '_' + str(args.b_num_nodes) + '_D_' + str_sim + name

for name in bash_list:
    new_path = currt_path + '/scripts/' + name
    print("Going to run " + new_path)
    os.system("sbatch " + new_path)

print("Done")