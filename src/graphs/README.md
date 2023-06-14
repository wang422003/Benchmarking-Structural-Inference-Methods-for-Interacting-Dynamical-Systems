# Generate Underlying Interaction Graphs

In this folder, we provide scripts to generate all eleven types of underlying interaction graphs mentioned in the paper. 

Following scripts are used to generated the corresponding graphs:

| Graph Type                                          | Script                                                       |
| --------------------------------------------------- | ------------------------------------------------------------ |
| Brain Networks (BN)                                 | generate_brain_networks_hierarchical.py                      |
| Chemical Reaction Networks in the Atmosphere (CRNA) | generate_chemical_reactions_in_atmosphere.py                 |
| Food Webs (FW)                                      | generate_food_webs.py                                        |
| Gene Coexpression Networks (GCN)                    | generate_gene_coexpression_networks.py                       |
| Gene Regulatory Networks (GRN)                      | [network_generation_algo](https://github.com/zhivkoplias/network_generation_algo)/src/test.py |
| Intercellular Networks (IN)                         | generate_intercellular_networks.py                           |
| Landscape Networks (LN)                             | generate_landscape_networks.py                               |
| Man-made Organic Reaction Networks (MMO)            | generate_man_made_organic_reaction_networks.py               |
| Reaction Networks inside Living Organisms (RNLO)    | generate_reaction_networks_inside_living_organism.py         |
| Social Networks (SN)                                | generate_social_networks_latest.py                           |
| Vascular Networks (VN)                              | generate_vascular_networks.py                                |

During generation, use args to generate the graphs fullfill the ranges of properties in Table 1 in Appendix.
