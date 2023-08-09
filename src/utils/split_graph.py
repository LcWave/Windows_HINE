import argparse
import random
import os


def split_dataset():
    args = init_para()
    edge_file = "../../dataset/" + args.dataset + "/edge.txt"
    train_link = []
    test_link = []
    with open(edge_file) as file:
        file = file.readlines()
        for line in file:
            token = line.strip('\n').split("\t")
            source_type, target_type = token[2].split('-')
            str = source_type + token[0] + ' ' + target_type + token[1]
            if random.random() < 0.8:
                train_link.append(str.strip())
            else:
                test_link.append(str.strip())

    d = "../../dataset/" + args.dataset

    train_file = os.path.join(d, '%s.train_0.8' % args.dataset)
    test_file = os.path.join(d, '%s.test_0.2' % args.dataset)

    train_str = '\n'.join(train_link)
    f = open(train_file, 'w', encoding='utf-8')
    f.write(train_str)
    f.close()

    test_str = '\n'.join(test_link)
    f = open(test_file, 'w', encoding='utf-8')
    f.write(test_str)
    f.close()


def init_para():
    parser = argparse.ArgumentParser(description="split_dataset")
    parser.add_argument('-d', '--dataset', default='yelp', type=str, help="Dataset")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    split_dataset()