import os
import argparse
import numpy as np
from collections import defaultdict
from src.model.transform_model import *

folder="./output/temp"
node_file="node.dat"
link_file="link.dat"
type_fil="type.dat"
baseinc_file="baseinc.dat"
allinc_file="allinc.dat"
emb_file=""

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-nodes', required=True, type=str, help='Input node file')
#     parser.add_argument('-links', required=True, type=str, help='Input link file')
#     parser.add_argument('-types', required=True, type=str, help='Input type file')
#     parser.add_argument('-output', required=True, type=str, help='Output embedding file')
#     parser.add_argument('-baseinc', required=True, type=str, help='Inconsistency of base aspects')
#     parser.add_argument('-allinc', required=True, type=str, help='Inconsistency of all aspects')
#     parser.add_argument('-binary', required=True, type=bool, help='Output format')
#     parser.add_argument('-size', required=True, type=int, help='Embedding dimension')
#     parser.add_argument('-negative', required=True, type=int, help='Negative sampling')
#     parser.add_argument('-samples', required=True, type=float, help='Training samples')
#     parser.add_argument('-alpha', required=True, type=float, help='Initial learning rate')
#     parser.add_argument('-threads', required=True, type=int, help='Number of working threads')
#     return parser.parse_args()"

def find_aspects(args, allinc, num_types, target_type):
    attributes, all_types = [], set()
    with open(f'{allinc}','r') as inc_file:
        for index, line in enumerate(inc_file):            
            if index%2==1: continue            
            line = line[:-1].split(' ')
            nodes, edges, node_start, edge_start = set(), set(), False, False
            for each in line:
                if each[0]=='N':
                    node_start = True
                    continue
                if each[0]=='E':
                    node_start = False
                    edge_start = True
                    continue
                if node_start:
                    each = each.replace('{','').replace('}','').replace('\'','').replace(',','')
                    nodes.add(int(each))
                if edge_start:
                    each = each.replace('{','').replace('}','').replace('\'','').replace(',','')
                    edges.add(int(each))

            if target_type in nodes:
                conflicts, add = [], True
                for i in range(len(attributes)):
                    if attributes[i][0].issubset(nodes) and attributes[i][0]!=nodes:
                        conflicts.append(i)
                    elif attributes[i][0]==nodes:
                        add = False

                for i in reversed(conflicts):
                    attributes.remove(attributes[i])
                if add:
                    attributes.append([nodes, edges])
                all_types = all_types.union(nodes)
                if len(all_types)==num_types:
                    break
                    
    return attributes


def concatenate(args, emb_file, attributes, target_type):

    all_embs, all_dims, all_nodes = [], [], set() 
    for i in range(len(attributes)):
        curr_emb = {}
        with open(f'{emb_file}.{i}', 'r') as file:
            for index, line in enumerate(file):
                if index<2: continue
                node, embs = line[:-1].split('\t')
                ntype, node = node.split(':')
                all_nodes.add(node)
                if index==2: 
                    all_dims.append(len(np.array(embs[:-1].split(' '))))
                curr_emb[node] = np.array(embs[:-1].split(' ')).astype(float)                
            all_embs.append(curr_emb)     
        os.remove(f'{emb_file}.{i}')
    
    final_embs = defaultdict(list)
    for curr_emb, dim in zip(all_embs, all_dims):
        for node in all_nodes:
            if node in curr_emb:
                final_embs[node].append(curr_emb[node])
            else:
                final_embs[node].append(np.zeros(dim))
    
    for each in final_embs:
        final_embs[each] = np.concatenate(final_embs[each])
        
    return final_embs
                
    
def write(args, count, emb_file, target_emb):

    with open(emb_file, 'w') as final_emb:
        final_emb.write(f'{len(target_emb)}\t{args.size}\n')
        for key, values in target_emb.items():
            flag = -1
            type = ""
            # print("key:",key)
            for node_type, node_count in count.items():
                # print(node_type, node_count)
                if int(key) >= node_count and flag < node_count:
                    flag = node_count
                    type = node_type
            key = type+str(int(key)-flag)
            final_emb.write(f'{key}\t')
            final_emb.write(' '.join(values.astype(str)))
            final_emb.write('\n')

def AspEm(config):
    args = config
    count = aspem_convert(config.target_node, config.data_type, config.relation_list, config.dataset)

    temp_folder = f"{folder}/{args.model}/{args.dataset}"
    nodes = f"{temp_folder}/{node_file}"
    links = f"{temp_folder}/{link_file}"
    types = f"{temp_folder}/{type_fil}"
    baseinc = f"{temp_folder}/{baseinc_file}"
    allinc = f"{temp_folder}/{allinc_file}"
    out_folder = f"./output/embedding/{args.model}"

    # create folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    emb_file = f"{out_folder}/{args.dataset}_{args.target_node}.txt"

    # statement = f'make -B src/model/AspEm'
    # os.system(statement)

    if not os.path.isfile(baseinc):
        print('Calculate Inconsistency of Base Aspects')
        statement = f'python src/model/AspEm/src/calc_inconsistency.py --input {links} --output {baseinc}'
        os.system(statement)
    
    if not os.path.isfile(allinc):
        print('Aggregate Inconsistency of All Aspects')
        statement = f'python src/model/AspEm/src/agg_inconsistency.py {baseinc} {allinc}'
        os.system(statement)
    
    num_types, target_type = 0, 0
    with open(types, 'r') as type_file:
        for index, line in enumerate(type_file):
            if index==0:
                target_type = int(line[:-1])
            elif index==1:
                num_types = int(line[:-1])
    
    attributes = find_aspects(args, allinc, num_types, target_type)
    print(attributes)
    
#     os.remove(f'{args.baseinc}')
#     os.remove(f'{args.allinc}')

    dims = np.full(len(attributes), int(50/len(attributes)))
    dims[:args.size%len(attributes)] += 1
    for index, (bach_nodes, bach_edges) in enumerate(attributes):
        
        bach_nodes.remove(target_type)
        attribute_nodes = ','.join(list(map(lambda x: str(x), bach_nodes)))
        attribute_edges = ','.join(list(map(lambda x: str(x), bach_edges)))
        
        print(f"Learning embedding for target node {target_type} with attribute nodes {attribute_nodes} and attribute edges {attribute_edges}")
        statement = f".\\src\\model\\Aspem\\bin\\aspem -center {target_type} -attribute {attribute_nodes} -edges {attribute_edges} -node {nodes} -hin {links} -output {emb_file}.{index} -binary {args.binary} -size {dims[index]} -negative {args.negative} -samples {args.samples} -alpha {args.alpha} -threads {args.threads}"
        os.system(statement)
        
    print(f'Concatenate embedding for target node {target_type}')
    target_emb = concatenate(args, emb_file, attributes, target_type)
    write(args, count, emb_file, target_emb)
    

# if __name__=='__main__':
#     main()