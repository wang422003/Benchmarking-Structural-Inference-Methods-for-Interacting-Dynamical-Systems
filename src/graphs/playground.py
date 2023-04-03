import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from generate_scale_free_n_small_world import save_graph, get_adjacency
import time
import sys
import powerlaw

np.set_printoptions(threshold=sys.maxsize)

li = []
li.append('./chemical_reaction_networks_in_atmosphere/ready/n250r3.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n250r1.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n250r2.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n50r1.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n50r2.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n100r3.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n50r3.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n100r1.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n50r2.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n50r3.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n100r2.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n150r1.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n150r2.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n150r3.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n200r1.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n200r2.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n200r3.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n15r1.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n15r2.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n15r3.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n30r1.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n30r2.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n30r3.graphml')
li.append('./chemical_reaction_networks_in_atmosphere/ready/n50r1.graphml')

# li.sort(key=lambda x: int(x.split('/')[-1].split('n')[1].split('r')[0]))

li = sorted(li, key= lambda x: (int(x.split('/')[-1].split('n')[1].split('r')[0]), int(x.split('/')[-1].split('n')[1].split('r')[1].split('.')[0])))
# print(li)

print(li[0].split('/', -1))

name_string = li[0]
name_ls = name_string.split('/')
length = len(name_ls)
new_name_st = ''
for i in range(length):
    if i < length - 1:
        new_name_st += name_ls[i] + '/'
    else:
        new_name_st += 'cache/' + name_ls[i].split('.')[0]
print(new_name_st)






