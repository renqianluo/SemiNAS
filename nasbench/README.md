# SemiNAS: Semi-Supervised Neural Architecture Search

This repository contains the code used for SemiNAS on NASBench-101 data set.


## Environments and Requirements
The code is built and tested on Pytorch 1.2

Install nasbench package and download NASBench-101 dataset. 

You can refer to nasbench/README.md or follow the insctructions below.

Install nasbench package from github (`https://github.com/google-research/nasbench.git`)

Download NASBench-101 dataset from https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
```
mkdir -p data
cd data
wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
cd ..
```

## Searching Architectures
To run the search process, please refer to `runs/train_seminas.sh`:
```
cd runs
bash train_seminas.sh
cd ..
```
After it finishes, it will report discovered architectures and corresponding performance.