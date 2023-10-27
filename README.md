![](https://img.shields.io/badge/PyG-GCN-red)
# PyG-GCN
* 包含已经有图结构的数据和无图结构的Knn构建方式

* PyG implementation of GCN (Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017).

* Datasets: CiteSeer, Cora, PubMed, NELL.

# Environment
pytorch==1.10.1+cu111

torch-geometric==2.0.3

# Usage
```python
python gcn.py
```

# Result
| dataset | Citeseer | Cora | Pubmed | NELL|
:-: | :-: | :-: | :-: | :-: |
|acc|58.40|65.90|77.80|66.25|
