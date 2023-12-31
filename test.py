import scipy.io as scio
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
import warnings
import argparse
import sys
import os
from src.utils.write_log import Logger


class evaluation:
    def __init__(self, seed, label_file, neg_2_file, neg_8_file, pos_2_file, pos_8_file):
        self.seed = seed
        self.node_label = {}
        self.label = {}
        la = []
        with open(label_file) as f:
            for line in f:
                i, l = line.strip().split()
                self.label[i] = int(l)
                la.append(int(l))
        self.n_label = len(set(la))

        # _____link_prediction______
        self.train_link_label = list()
        self.test_link_label = list()
        self.sample_8 = list()
        self.sample_2 = list()
        with open(pos_2_file) as infile:
            for line in infile.readlines():
                u, b = [item for item in line.strip().split()]
                self.test_link_label.append([u, b, 1])

        with open(pos_8_file) as infile:
            for line in infile.readlines():
                u, b = [item for item in line.strip().split()]
                self.train_link_label.append([u, b, 1])

        with open(neg_8_file) as infile:
            for line in infile.readlines():
                u, b, label = [item for item in line.strip().split()]
                self.sample_8.append([u, b, int(label)])

        with open(neg_2_file) as infile:
            for line in infile.readlines():
                u, b, label = [item for item in line.strip().split()]
                self.sample_2.append([u, b, int(label)])

    def evaluate_cluster(self, embedding_list):
        X = []
        Y = []
        for p in self.label:
            X.append(embedding_list[p])
            Y.append(self.label[p])

        Y_pred = KMeans(self.n_label, random_state=self.seed).fit(np.array(X)).predict(X)
        nmi = normalized_mutual_info_score(np.array(Y), Y_pred)
        ari = adjusted_rand_score(np.array(Y), Y_pred)
        return nmi, ari


    def evaluate_clf(self, embedding_list):
        X = []
        Y = []
        for p in self.label:
            X.append(embedding_list[p])
            Y.append(self.label[p])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)

        LR = LogisticRegression()
        LR.fit(X_train, Y_train)
        Y_pred = LR.predict(X_test)

        micro_f1 = f1_score(Y_test, Y_pred, average='micro')
        macro_f1 = f1_score(Y_test, Y_pred, average='macro')
        return micro_f1, macro_f1

    def link_prediction(self, embedding_matrix):

        # score = self.evaluate_business_cluster(embedding_matrix)
        # print 'nmi = ', score

        train_x = []
        train_y = []
        for u, b, label in self.train_link_label:
            if(u in embedding_list and b in embedding_list):
                train_x.append(embedding_list[u] + embedding_list[b])
                train_y.append(float(label))
        for u, b, label in self.sample_8:
            if (u in embedding_list and b in embedding_list):
                train_x.append(embedding_list[u] + embedding_list[b])
                train_y.append(float(label))

        test_x = []
        test_y = []
        for u, b, label in self.test_link_label:
            if (u in embedding_list and b in embedding_list):
                test_x.append(embedding_list[u] + embedding_list[b])
                test_y.append(float(label))
        for u, b, label in self.sample_2:
            if (u in embedding_list and b in embedding_list):
                test_x.append(embedding_list[u] + embedding_list[b])
                test_y.append(float(label))

        lr = LogisticRegression(max_iter=300)
        lr.fit(train_x, train_y)

        pred_y = lr.predict_proba(test_x)[:, 1]
        pred_label = lr.predict(test_x)
        test_y = np.array(test_y)

        auc = roc_auc_score(test_y, pred_y)
        f1 = f1_score(test_y, pred_label, average='macro')
        acc = accuracy_score(test_y, pred_label)

        return auc, f1, acc





def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def load_emd(filename):
    embedding_matrix = {}
    i = 0
    with open(filename) as infile:
        line = infile.readline().strip().split()
        n = int(line[0])
        for line in infile:
            emd = line.strip().split()
            if emd[1] != "nan":
                embedding_matrix[emd[0]] = str_list_to_float(emd[1:])
            else:
                print("nan error!")
            i = i + 1
        if(i != n):
            print("number of nodes error!")
    return embedding_matrix
parser = argparse.ArgumentParser(description="test")
parser.add_argument('-d', '--dataset', default='dblp', type=str, help="Dataset")
parser.add_argument('-m', '--model', default='MetaGraph2vec', help='Train model')
parser.add_argument('-n', '--name', default='dblp_node.txt', type=str, help='Evaluation task')
parser.add_argument('-s', '--seed', default=0, type=str, help='seed')
args = parser.parse_args()

log_path = './Logs/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置

    # 记录正常的 print 信息
sys.stdout = Logger(log_path, args, 'test')
# 记录 traceback 异常信息
sys.stderr = Logger(log_path, args, 'test')

filename = "./output/embedding/" + args.model + "/" + args.name
label_file = "./dataset/" + args.dataset + "/label.txt"
neg_2_file = "./dataset/" + args.dataset + "/neg_0.2"
neg_8_file = "./dataset/" + args.dataset + "/neg_0.8"
pos_2_file = "./dataset/" + args.dataset + "/" + args.dataset + ".test_0.2"
pos_8_file = "./dataset/" + args.dataset + "/" + args.dataset + ".train_0.8"


_evaluation = evaluation(args.seed, label_file, neg_2_file, neg_8_file, pos_2_file, pos_8_file)
embedding_list = load_emd(filename)
NMI, ARI = _evaluation.evaluate_cluster(embedding_list)
micro, macro = _evaluation.evaluate_clf(embedding_list)
auc, f1, acc = _evaluation.link_prediction(embedding_list)



print('<Cluster>        NMI = %.4f, ARI = %.4f' % (NMI, ARI))

print('<Classification>     Micro-F1 = %.4f, Macro-F1 = %.4f' % (micro, macro))

print('<link_prediction>     auc = %.4f, f1 = %.4f, acc = %.4f' % (auc, f1, acc))
# print('<Cluster> 		NMI = %.4f, ARI = %.4f(%.4f)' % (
#     np.mean(_NMI), np.mean(_ARI), np.std(_ARI, ddof=1)))

# print('<Classification> 	Micro-F1 = %.4f(%.4f), Macro-F1 = %.4f(%.4f)' % (
#     np.mean(_micro), np.std(_micro, ddof=1), np.mean(_macro), np.std(_macro, ddof=1)))
