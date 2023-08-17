import numpy as np
import os
from collections import defaultdict
data_folder, output_folder = '../dataset', '../output/temp'
node_file, link_file, label_file = 'node.dat', 'link.dat', 'label.dat'
info_file, meta_file = 'info.dat', 'meta.dat'

def hgt_convert(target_type, data_type, nlabel, relation_list, dataset, attributed, supervised):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{output_folder}/HGT/{dataset}'
    # ori_data_folder = f'{data_folder}/{dataset}'
    # model_data_folder = f'{model_folder}/HGT/data/{dataset}'

    # create folder
    if not os.path.exists(model_data_folder):
        os.makedirs(model_data_folder)

    print('HGT: converting {}\'s node file for {} training!'.format(dataset,
                                                                    'attributed' if attributed == 'True' else 'unattributed'))
    type_dict = defaultdict(set)
    dictionary = {}
    add_account = {}

    # 建立字典
    i = 0
    for e in data_type:
        dictionary[e] = i
        add_account[e] = i
        i += 1

    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    with open(f'{ori_data_folder}/edge.txt', 'r') as original_node_file:
        for line in original_node_file:
            start, end, edge_type, weight = line[:-1].split('\t')
            start_type, end_type = edge_type.split('-')
            type_dict[start_type].add(start)
            type_dict[end_type].add(end)

        addition = 0
        type_to_ids = defaultdict(set)
        for node_type, original_set in type_dict.items():
            new_set = {int(x) + addition for x in original_set}
            type_to_ids[node_type] = new_set
            add_account[node_type] = addition
            addition += len(new_set)
        # print(type_to_ids)
        # print(add_account)
        for node_type, original_set in type_to_ids.items():
            for node in original_set:
                if attributed == True:
                    new_node_file.write(f'{node}\t{dictionary[node_type]}\t{line[3]}\n')
                elif attributed == False:
                    new_node_file.write(f'{node}\t{dictionary[node_type]}\n')
                    # print(node, dictionary[node_type])
        new_node_file.close()

    links = relation_list.split('+')
    next_value = 0
    new_dict = {}
    for item in links:
        new_dict[item] = next_value
        next_value += 1

    print(f'HGT: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/edge.txt', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line.strip().split('\t')
            left_type, right_type = ltype.split('-')
            new_link_file.write(f'{int(left) + add_account[left_type]}\t{int(right) + add_account[right_type]}\t{new_dict[ltype]}\n')
    new_link_file.close()

    if supervised == True:
        print(f'HGT: converting {dataset}\'s label file for semi-supervised training!')
        # labeled_type, nlabel, begin = None, -1, False
        labeled_type = dictionary[target_type]
        nlabel = nlabel
        # with open(f'{ori_data_folder}/{info_file}', 'r') as file:
        #     for line in file:
        #         if line.startswith('Targeting: Label Type'):
        #             labeled_type = int(line.split(' ')[-1])
        #         elif line == 'TYPE\tCLASS\tMEANING\n':
        #             begin = True
        #         elif begin:
        #             nlabel += 1
        new_label_file = open(f'{model_data_folder}/{label_file}', 'w')
        new_label_file.write(f'{labeled_type}\t{nlabel}\n')
        with open(f'{ori_data_folder}/label.txt', 'r') as original_label_file:
            for line in original_label_file:
                line = line[:-1].split('\t')
                line[0] = int(line[0][1:]) + add_account[line[0][0]]
                new_label_file.write(f'{line[0]}\t{line[1]}\n')
        new_label_file.close()

    return add_account
