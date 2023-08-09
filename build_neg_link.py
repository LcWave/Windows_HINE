from src.config import Config
from src.utils.data_process import *
import argparse
from collections import defaultdict
import numpy as np
import os


def build_graph():
    args = init_para()
    edge_file = "./dataset/" + args.dataset + "/edge.txt"
    graph = defaultdict(list)
    with open(edge_file) as file:
        file = file.readlines()
        for line in file:
            token = line.strip('\n').split("\t")
            source_type, target_type = token[2].split('-')
            graph[source_type + token[0]].append(target_type + token[1])
    return graph


def build_neg_link():
    args = init_para()
    config_file = ["./src/config.ini"]
    config = Config(config_file, args)

    g_hin = HIN(config.input_fold, config.data_type, config.relation_list)
    M_id = reverse_dict(g_hin.node2id_dict)
    graph = build_graph()


    sample_8 = []
    sample_2 = []
    for i in range(config.link_size):

        a = np.random.randint(config.node_size)
        b = np.random.randint(config.node_size)
        while(M_id[b] in graph[M_id[a]]):
            a = np.random.randint(config.node_size)
            b = np.random.randint(config.node_size)
        c = np.random.rand()
        if(c <= 0.8):
            sample_8.append(M_id[a] + ' ' + M_id[b] + ' ' + '0')
        else:
            sample_2.append(M_id[a] + ' ' + M_id[b] + ' ' + '0')

    d = "./dataset/" + args.dataset

    train_file = os.path.join(d, 'neg_0.8')
    test_file = os.path.join(d, 'neg_0.2')

    train_str = '\n'.join(sample_8)
    f = open(train_file, 'w', encoding='utf-8')
    f.write(train_str)
    f.close()

    test_str = '\n'.join(sample_2)
    f = open(test_file, 'w', encoding='utf-8')
    f.write(test_str)
    f.close()




def reverse_dict(dict):
    dict_new = {}
    for k, v in dict.items():
        dict_new[v] = k
    return dict_new


def init_para():
    parser = argparse.ArgumentParser(description="OPEN-HINE")
    parser.add_argument('-d', '--dataset', default='yelp', type=str, help="Dataset")
    parser.add_argument('-m', '--model', default='MetaGraph2vec', type=str, help='Train model')
    # parser.add_argument('-t', '--task', default='node_classification', type=str, help='Evaluation task')
    # parser.add_argument('-p', '--metapath', default='pap', type=str, help='Metapath sampling')
    # parser.add_argument('-s', '--save', default='1', type=str, help='save temproal')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    build_neg_link()
