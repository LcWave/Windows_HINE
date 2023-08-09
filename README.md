# Windows_HINE

This is an open-source toolkit for Heterogeneous Information Network Embedding under Windows (Windows_HINE) with version 0.1. We can train and test the model more easily. It provides implementations of many popular models.

We reference the [OpenHINE](https://github.com/BUPT-GAMMA/OpenHINE) and the [HNE](https://github.com/yangji9181/HNE).


## Get started

### Requirements and Installation

- Python version >= 3.6

- PyTorch version >= 1.4.0

- TensorFlow version  >= 1.14

- Keras version >= 2.3.1


### config/Usage

##### Input parameter

```python
python train.py -m model_name -d dataset_name
```

e.g.

```python
python train.py -m Metapath2vec -d acm
```



##### Model Setup

The model parameter could be modified in the file ( ./src/config.ini ).

- ###### 	common parameter


​	--alpha:	learning rate

​	--dim:	dimension of output

​	--epoch: the number of iterations	

​	--num_workers:number of workers for dataset loading (It should be set to 0, if you are in trouble with Windows OS.)

​	etc...

- ###### 	specific parameter


​	--metapath:	the metapath selected

​	--neg_num:	the number of negative samples		

​	etc...

### Datasets

If you want to train your own dataset, create the file (./dataset/your_dataset_name/edge.txt) and the format   is as follows：

###### 	input:	edge

​		src_node_id	dst_node_id	edge_type	weight

​	e.g.

		19	7	p-c	2
		19	7	p-a	1
		11	0	p-c	1
		0	11	c-p	1

PS：The input graph is directed and the undirected needs to be transformed into directed graph. And all types of node should start from 0.

###### Input:	label	


​		node_name node_label

e.g.

```
p0	0
p1	0
p2	1
p3	1
p4	2
p5	2
... ...
```



## Model

#### Available

##### [HeGAN KDD 2019]

​		Adversarial Learning on Heterogeneous Information Network 

​		src code:https://github.com/librahu/HeGAN

##### 	[HERec TKDE 2018]

​		Heterogeneous Information Network Embedding for Recommendation 

​		src code:https://github.com/librahu/HERec

###### 		*spec para: 

​			metapath_list: pap|psp	(split by "|")

##### 	[HIN2Vec CIKM 2017]

​		HIN2Vec: Explore Meta-paths in Heterogeneous Information Networks for Representation Learning 

​		src code:https://github.com/csiesheep/hin2vec

##### 	[metapath2vec KDD 2017]

​		metapath2vec: Scalable Representation Learning for Heterogeneous Networks 

​		src code:https://ericdongyx.github.io/metapath2vec/m2v.html

​		the python version implemented by DGL:https://github.com/dmlc/dgl/tree/master/examples/pytorch/metapath2vec 

##### 	[MetaGraph2Vec PAKDD 2018]

​		MetaGraph2Vec: Complex Semantic Path Augmented Heterogeneous Network Embedding

​		src code:https://github.com/daokunzhang/MetaGraph2Vec

## Output

#### Test

```python
python test.py -d dataset_name -m model_name -n file_name
```

The output embedding file name can be found in (./output/embedding/model_name/) .

e.g.

```python
python test.py -d dblp -m HAN -n node.txt
```

###### output:	embedding	

​		number_of_nodes embedding_dim

​		node_name dim1 dim2 

e.g.

```
11246	2
a1814 0.06386946886777878 -0.04781734198331833
a0 ... ...
```