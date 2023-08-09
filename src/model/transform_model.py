import numpy as np
import os
from collections import defaultdict
import networkx as nx
data_folder, output_folder = './dataset', './output/temp'
node_file, link_file, label_file = 'node.dat', 'link.dat', 'label.dat'
info_file, meta_file = 'info.dat', 'meta.dat'


def ckd_link_convert(targetnode, data_type, relation_list, dataset, attributed,version,use_target_node):
    """
    依据不同的连接方式,切分成多个子图
    version:实验版本.
    use_target_node:是否要默认添加target_node之间的连接
    """

    def add_node(neigh_map,start_node,end_node,type_map,all_node_types):
        if start_node not in neigh_map:
            neigh_map[start_node]={}
            for node_type in all_node_types:
                neigh_map[start_node][node_type]=set()
        if end_node not in neigh_map:
            neigh_map[end_node]={}
            for node_type in all_node_types:
                neigh_map[end_node][node_type]=set()
        neigh_map[start_node][type_map[end_node]].add(end_node)
        neigh_map[end_node][type_map[start_node]].add(start_node)

    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{output_folder}/CKD/{dataset}/{version}'

    dictionary = {}
    add_count = {}
    node_type_map={}#node->node_type
    node_neigh_type_map={}#node->node_type->neigh_node
    node_types=set()#节点类型的集合
    target_node_set=set()#目标结点的集合
    node2id = {}#目标节点转成新的idx
    useful_types=[]

    print(f'CKD: writing {dataset}\'s config file!')
    target_node, target_edge, ltypes = 0, 0, []

    #建立字典
    i = 0
    for e in data_type:
        dictionary[e] = i
        add_count[e] = i
        i += 1
    # print(dictionary)

    target_node = dictionary[targetnode]
    target_edge = 0

    # if dataset == "acm":
    #     target_node = 1
    #     target_edge = 0
    # elif dataset == "dblp":
    #     target_node = 0
    #     target_edge = 0
    # elif dataset == "yelp":
    #     target_node = 0
    #     target_edge = 0

    links = relation_list.split('+')
    next_value = 0
    new_dict = {}
    for item in links:
        new_dict[item] = next_value
        ltype = str(next_value)
        next_value += 1
        nodes = item.split('-')
        snode = str(dictionary[nodes[0]])
        enode = str(dictionary[nodes[1]])
        ltypes.append((snode, enode, ltype))


    # create folder
    if not os.path.exists(model_data_folder):
        os.makedirs(model_data_folder)

    config_file = open(f'{model_data_folder}/config.dat','w')
    config_file.write(f'{target_node}\n')
    config_file.write(f'{target_edge}\n')
    config_file.write('{}\n'.format('\t'.join(list(map(lambda x: ','.join(x), ltypes)))))
    config_file.close()

    print('CKD Link: converting {}\'s node file for {} training!'.format(dataset, 'unattributed'))
    node_type_ids = defaultdict(set)
    # with open(f'../../dataset/acm/edge.txt', 'r') as file:
    with open(f'{ori_data_folder}/edge.txt', 'r') as file:
        for line in file:
            parts = line.split('\t')
            start_node = int(parts[0])
            end_node = int(parts[1])
            link_types = parts[2]
            link_type = link_types.split("-")

            node_type_ids[link_type[0]].add(start_node)
            node_type_ids[link_type[1]].add(end_node)
    addition = 0
    type_to_ids = defaultdict(set)
    for node_type, original_set in node_type_ids.items():
        new_set = {x + addition for x in original_set}
        type_to_ids[node_type] = new_set
        add_count[node_type] = addition
        addition += len(new_set)


    new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
    for node_type, original_set in type_to_ids.items():
        type = dictionary[node_type]
        node_types.add(int(type))
        for item in original_set:
            new_node_file.write(f'{item}\t{type}\n')
            node_type_map[int(item)] = int(type)
            if int(type) == target_node:
                node2id[int(item)] = len(node2id)
                target_node_set.add(int(item))
    new_node_file.close()

    print(f'CKD Link: converting {dataset}\'s label file')
    new_label_file = open(f'{model_data_folder}/{label_file}', 'w')
    with open(f'{ori_data_folder}/label.txt', 'r') as original_label_file:
        for line in original_label_file:
            line = line[:-1].split('\t')
            line[0] = line[0][1:]
            new_label_file.write(f'{line[0]}\t{line[1]}\n')
    new_label_file.close()

    type_corners = {int(ltype[2]): defaultdict(set) for ltype in ltypes}

    print(f'CKD: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/edge.txt', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            left = int(left) + add_count[ltype[0]]
            right = int(right) + add_count[ltype[2]]
            ltype = new_dict[ltype]
            new_link_file.write(f'{left}\t{right}\t{ltype}\n')
            #add_node(node_neigh_type_map,int(left),int(right),node_type_map,node_types)
            # origin_graph.add_edge(int(left), int(right), weight=int(weight), ltype=int(ltype),
            #                direction=1 if left <= right else -1)
            start, end, ltype = int(left), int(right), int(ltype)
            if start in node2id:
                type_corners[ltype][end].add(node2id[start])
            if end in node2id:
                type_corners[ltype][start].add(node2id[end])
    new_link_file.close()


    #get homogeneous graph
    for ltype in ltypes:
        if int(ltype[0])==target_node or int(ltype[1])==target_node:
            useful_types.append(int(ltype[2]))

    for ltype in useful_types:
        # if dataset=='DBLP2' and ltype==2:
        #     continue
        corners = type_corners[ltype]
        #根据同一个start node,从而判断节点之间的二阶关系
        two_hops = defaultdict(set)
        graph=nx.Graph(node_type=int)
        for _, neighbors in corners.items():
            #print(f'ltype:{ltype},node_cnt:{len(neighbors)}')
            for snode in neighbors:
                for enode in neighbors:
                    if snode!=enode:
                        #two_hops[snode].add(enode)
                        graph.add_edge(snode,enode)
        #如果缺少边,则添加自环
        for node in node2id.values():
            if node not in graph:
                graph.add_edge(node,node)
        print(f'write graph {ltype},node:{len(graph.nodes)},edge:{len(graph.edges)}')
        nx.write_edgelist(graph,path=f"{model_data_folder}/sub_graph_{ltype}.edgelist",delimiter='\t',data=False)

    # #原始图的一阶关系
    # for ltype in ltypes:
    #     snode,enode,l_type=[int(i) for i in ltype]
    #     if snode==target_node and enode==target_node and l_type==target_edge:
    #         graph=nx.Graph(node_type=int)
    #         corners = type_corners[l_type]
    #         for origin_node,neighbors in corners.items():
    #             new_node_id=node2id[origin_node]
    #             for nei in neighbors:
    #                 graph.add_edge(new_node_id,nei)
    #         for node in node2id.values():
    #             if node not in graph:
    #                 graph.add_edge(node,node)
    #         print(f'write graph origin,node:{len(graph.nodes)},edge:{len(graph.edges)}')
    #         nx.write_edgelist(graph, path=f"{model_data_folder}/sub_graph_origin.edgelist", delimiter='\t', data=False)

    #add node to new_id map file
    with open(f"{model_data_folder}/node2id.txt",'w') as f:
        for node,id in node2id.items():
            f.write('\t'.join([str(node),str(id)])+'\n')


def aspem_convert(target_type, data_type, relation_list, dataset):
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{output_folder}/AspEm/{dataset}'

    print(f'AspEm: converting {dataset}\'s node file!')
    type_dict = defaultdict(set)
    dictionary = {}
    add_account = {}

    # 建立字典
    i = 0
    for e in data_type:
        dictionary[e] = i
        add_account[e] = i
        i += 1

    # create folder
    if not os.path.exists(model_data_folder):
        os.makedirs(model_data_folder)

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
        # print(add_account)
        for node_type, original_set in type_to_ids.items():
            for node in original_set:
                new_node_file.write(f'{dictionary[node_type]}:{node} {dictionary[node_type]}\n')
        new_node_file.close()

    links = relation_list.split('+')
    next_value = 0
    new_dict = {}
    for item in links:
        new_dict[item] = next_value
        next_value += 1

    print(f'AspEm: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
    with open(f'{ori_data_folder}/edge.txt', 'r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line.strip().split('\t')
            left_type, right_type = ltype.split('-')
            new_link_file.write(
                f'{dictionary[left_type]}:{int(left) + add_account[left_type]} {dictionary[left_type]} {dictionary[right_type]}:{int(right) + add_account[right_type]} {dictionary[right_type]} {weight} {new_dict[ltype]}\n')
    new_link_file.close()

    print(f'AspEm: writing {dataset}\'s type file!')
    type_count = len(dictionary)
    target_type = dictionary[target_type]
    # type_count, target_type = 0, -1
    # with open(f'{ori_data_folder}/{meta_file}', 'r') as original_meta_file:
    #     for line in original_meta_file:
    #         entity, info, _, _ = line[:-1].split(' ')
    #         info = info[:-1].split('_')
    #         if entity == 'Node' and info[0] == 'Type':
    #             type_count += 1
    #         if entity == 'Label' and info[0] == 'Class':
    #             target_type = info[1]
    #             break
    new_type_file = open(f'{model_data_folder}/type.dat', 'w')
    new_type_file.write(f'{target_type}\n')
    new_type_file.write(f'{type_count}\n')
    new_type_file.close()

    return add_account