import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from ogb.graphproppred import DglGraphPropPredDataset
import periodictable
from tqdm import tqdm

chirality_dict = {
    0: "unspecified chirality",
    1: "clockwise tetrahedral chirality",
    2: "counterclockwise tetrahedral chirality",
    3: "chirality type that does not fit the standard tetrahedral classification",
    4: "miscellaneous chirality"
}
number_dict = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
    10: 'ten'
    
}
Hybridization_dict = {
    0 : "SP",
    1 : "SP2",
    2 : "SP3",
    3 : "SP3D",
    4 : "SP3D2",
    5 : "unknown"}
bone_type_dict = {
    0: 'single',
    1: 'double',
    2: 'triple',
    3: 'aromatic',
    4: 'unknown'
}
bond_stereochemistry_dict = {
    0: 'STEREONONE',
    1: 'STEREOZ',
    2: 'STEREOE',
    3: 'STEREOCIS',
    4: 'STEREOTRANS',
    5: 'STEREOANY',
}
root_path = 'your_local_root_path'
dataset_name = 'molbbbp'
if __name__ == "__main__":
    device = torch.device("cuda:0")
    PLM  = SentenceTransformer('all-mpnet-base-v2').to(device)
    datasets = DglGraphPropPredDataset(name='ogbg-{}'.format(dataset_name),root=root_path)
    map_index = {}
    over_all_idx = 0
    dp_list = []
    with tqdm(range(len(datasets))) as pbar:
        for i in pbar:
            graph = datasets[i][0]
            for j in range(graph.num_nodes()):
                '''
                    could go through ogb.ogb.utils.feature to check feature extraction
                    node feat 9-dims:
                        0. Atomic number,
                        1. Chirality (0 ~ 3),
                        2. Degree ( graph node degree),
                        3. Formal charge (integer, could be negative),
                        4. number of total Hs (it may be implicit, integer),
                        5. Number of radical electrons (non-negative integer),
                        6. Hybridization (0 ~ 8),
                        7. Aromaticity (binary 1 means part of an aromatic ring),
                        8. Ring (binary 1 means part of a ring)
                '''
                current_node_feat = graph.ndata['feat'][j]
                feat_key  = tuple(current_node_feat.tolist())
                if feat_key in map_index.keys():
                    continue
                map_index[feat_key] = over_all_idx
                over_all_idx += 1
                element_name = periodictable.elements[feat_key[0]].name
                chirality_str = chirality_dict[feat_key[1]]
                degree_str = number_dict[feat_key[2]]
                formal_charge_str = str(feat_key[3])
                number_of_h =  'unknown' if feat_key[4] > 8 else number_dict[feat_key[4]]
                number_of_radical_e = 'unknown' if feat_key[5] > 4 else number_dict[feat_key[5]]
                hybridization = Hybridization_dict[feat_key[6]]
                node_description = "This atom is {}. It has a {}. Its degree is {}. Its formal charge is {}. The radical electrons of this atom is {}. Its hybridization type is {}.".format(element_name, chirality_str, degree_str, formal_charge_str, number_of_radical_e, hybridization)
                if number_of_h != 'unknown':
                    node_description += " It connects {} hydrogen atoms.".format(number_of_h)
                if feat_key[7] == 1:
                    node_description += " This atom is part of an aromatic ring."
                if feat_key[8] == 1:
                    node_description += " This atom is part of a ring."
                dp = {
                        "embedding":PLM.encode(node_description,convert_to_tensor=True).cpu(), 
                        "label":node_description,
                        "element_name": element_name,
                        "chirality": chirality_str,
                        "formal_charge": formal_charge_str,
                        "number_of_h": number_of_h,
                        "number_of_radical_e": number_of_radical_e,
                        "hybridization": hybridization,
                        "aromatic_ring": feat_key[7],
                        "ring": feat_key[8],
                        'degree': degree_str,
                    }
                dp_list.append(dp)
            for j in range(graph.num_edges()):
                '''
                could go through ogb.ogb.utils.feature to check feature extraction
                    edge feat 3-dims:
                        0. Bond type
                        1. Bond stereo
                        3. is conjuageted 
                '''
                curr_edge_feature = graph.edata['feat'][j]
                feat_key = tuple(curr_edge_feature.tolist())
                if feat_key in map_index.keys():
                    continue
                map_index[feat_key] = over_all_idx
                over_all_idx += 1
                edge_description = "This is a chemistry bond. Its bond type is {}. Its bond stereochemistry is {}.".format(bone_type_dict[feat_key[0]], bond_stereochemistry_dict[feat_key[1]])
                if feat_key[2] == 1:
                    edge_description += ' This bond is conjugated.'
                else:
                    edge_description += ' This bond is isolated.'
                dp = {
                        'embedding': PLM.encode(edge_description,convert_to_tensor=True).cpu(), 
                        "label":edge_description,
                        "bond_type": bone_type_dict[feat_key[0]],
                        "bond_stereochemistry": bond_stereochemistry_dict[feat_key[1]],
                        "conjugated": feat_key[2]
                    }
                dp_list.append(dp)
    print("overall number: ",over_all_idx)
    torch.save(dp_list,root_path+'/ogbg_{}/text_to_embedding.bin'.format(dataset_name))
    torch.save(map_index,root_path+'/ogbg_{}/map_index_vector_to_text.bin'.format(dataset_name))