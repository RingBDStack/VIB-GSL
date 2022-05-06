# VIB-GSL
Code for "Graph Structure Learning with Variational Information Bottleneck" (submitted to AAAI 2022).

## Overview
- main.py: getting started
- param_parser.py: default paramater setting.
- train_eval.py: k-fold cross-validation for model training and evaluation. 
- backbone.py: the basic GNNs we used as the backbone, including GCN, GAT and GIN.
- gsl.py: overall framewprk of VIB-GSL.
- layers.py: graph learner of IB-Graph.
- utils.py: data preprocessing and loading.

## Requirements
The implementation of VIB-GSL is tested under Python 3.6.7, with the following packages installed:
* `numpy==1.19.2`
* `torch==1.7.0`
* `torch-cluster==1.5.9`
* `torch-geometric==1.6.3`
* `torch-scatter==2.0.6`
* `torch-sparse==0.6.9`

## Datasets
All the datasets (i.e., IMDB-B, IMDB-M, REDDIT-B, COLLAB) are provided by [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.TUDataset). 

## Run the codes
Data preprocessing:  
The implemention of data preprocessing is modified based on [this](https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/datasets.py).  
Train and evaluate the model:
* `python main.py --dataset_name <dataset> --backbone <backbone>`  

We train the VIB-GSL with GNN backbone, and report the `training loss`, `validation loss`, `validation accuracy` and the `test accuracy`.  
For instance:
* `python main.py --dataset_name IMDB-BINARY --backbone GCN` 
```
Epoch: 10, train loss: 0.947, train acc: 0.655, val loss: 0.96006, val acc: 0.680, test scc: 0.690
Epoch: 20, train loss: 0.820, train acc: 0.724, val loss: 0.88318, val acc: 0.660, test scc: 0.660
Epoch: 30, train loss: 0.770, train acc: 0.736, val loss: 0.84559, val acc: 0.690, test scc: 0.720
Epoch: 40, train loss: 0.782, train acc: 0.752, val loss: 0.83357, val acc: 0.690, test scc: 0.730
Epoch: 50, train loss: 0.762, train acc: 0.726, val loss: 0.82110, val acc: 0.690, test scc: 0.740
Epoch: 60, train loss: 0.754, train acc: 0.745, val loss: 0.81510, val acc: 0.700, test scc: 0.740
Epoch: 70, train loss: 0.753, train acc: 0.730, val loss: 0.82457, val acc: 0.690, test scc: 0.690
Epoch: 80, train loss: 0.769, train acc: 0.731, val loss: 0.80603, val acc: 0.680, test scc: 0.750
Epoch: 90, train loss: 0.730, train acc: 0.749, val loss: 0.82343, val acc: 0.670, test scc: 0.750
Epoch: 100, train loss: 0.730, train acc: 0.735, val loss: 0.82211, val acc: 0.700, test scc: 0.750
...
Fold: 0, train acc: 0.745, Val loss: 0.775, Val acc: 0.75000, Test acc: 0.750
```

Load the trained model and get test accuracy:
* `python train_eval.py --dataset_name <dataset> --backbone <backbone>` 
For instance:
* `python train_eval.py --dataset_name IMDB-BINARY --backbone GCN`
Import "results/IMDB-BINARY_GCN.pth" into the model, you will get:
```
Test acc of IMDB-BINARY_GCN is: 0.74
```


