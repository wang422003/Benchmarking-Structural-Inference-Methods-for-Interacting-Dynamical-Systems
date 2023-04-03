import os

currt_path = os.getcwd()

bash_list = [
    'train_ACD_CRN_30_D_NS_R1.sh',
    'train_ACD_CRN_30_D_NS_R2.sh',
    'train_ACD_CRN_30_D_NS_R3.sh',
    'train_ACD_CRN_30_D_NS_R1_seed10.sh',
    'train_ACD_CRN_30_D_NS_R2_seed10.sh',
    'train_ACD_CRN_30_D_NS_R3_seed10.sh',
    'train_ACD_CRN_30_D_NS_R1_seed144.sh',
    'train_ACD_CRN_30_D_NS_R2_seed144.sh',
    'train_ACD_CRN_30_D_NS_R3_seed144.sh',
]

for name in bash_list:
    new_path = currt_path + '/scripts/' + name
    print("Going to run " + new_path)
    os.system("sbatch " + new_path)

print("Done")