# MFF-nDA: : A General System Framework for ncRNA-Disease Association Prediction Based on Multi-module Fusion

MFF-nDA is a deep learning model for predicting some ncRNA-disease associations.

![Alt text](figure/flowchart.jpg?raw=true "MFF-nDA pipeline")

# Getting Started

## Installation

Setup conda environment:
```
conda create -n MFF-nDA python=3.10 -y
conda activate MFF-nDA
```

Install required packages
```
pip install numpy==1.25.0
pip install scipy==1.11.1
pip install pandas==1.5.3
pip install openpyxl==3.0.10
pip install scikit-learn==1.2.2
pip install biopython==1.83
pip install obonet==1.0.0
pip install gensim==4.3.1
pip install tqdm==4.65.0
pip install matplotlib==3.8.2
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_cluster-1.6.2%2Bpt21cu121-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_scatter-2.1.2%2Bpt21cu121-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_sparse-0.6.18%2Bpt21cu121-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_spline_conv-1.2.2%2Bpt21cu121-cp310-cp310-win_amd64.whl
pip install torch-geometric
```

# Prepare

[Click the link to download all code and data](https://1drv.ms/f/c/9f80329bc8ba4280/EvM7tdJ9-KREsyrqcpm-vWgBPhddPQ4Ss1NoiKLUjjeqkg?e=T0eegm)

```
${DATA_ROOT}
├── MFF-nDA
│   ├── data
│   │   ├──adj.csv *(adjacent matrix)*
│   │   ├──ncRNA_sim_3.txt
│   │   │   ├──seq_sim.txt
│   │   │   ├──Func_sim.txt
│   │   │   ├──GIPKs.txt
│   │   ├──disease_sim_2.txt
│   │   │   ├──doid_sim.txt
│   │   │   ├──GIPKd.txt
│   ├── MSNF
│   │   ├──heterogeneous_3_2.txt
│   ├── MFF
│   │   ├──Transformer
│   │   │   ├──trans_feat.npy
│   │   ├──GCN
│   │   │   ├──gcn_feat.npy
│   │   ├──GAT
│   │   │   ├──gat_feat.npy
│   ├── model.py
│   ├── utils.py
│   ├── metrics.py
│   ├── main.py

```

