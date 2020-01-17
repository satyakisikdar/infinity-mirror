#   notes on Graph Auto-Encoders
- can choose between the following two models:
    - `gcn_ae` is a Graph Auto-Encoder with the `GCN` encoder
    - `gcn_vae` is a Variational Graph Auto-Encoder with the `GCN` encoder
- to use your own data, provide:
    - (required) `N × N` adjacency matrix, where `N` is the number of nodes in the graph
    - (optional) `N × D` feature matrix, where `D` is the number of features per node
- to enable GPU, use the `--nv` singularity flag
- remember to make sure that `gpu_id` is always set to a sensible value
    - e.g., don't have `gpu_id=3` if there is only one GPU on the system configured with CUDA
    - if there is only one GPU configured, set `gpu_id=0` wherever encountered

##  requirements
### conda
- `python 2.7`
    - latest version of `Miniconda`
- `pip 19`
    - make sure to upgrade `pip` using `pip install --upgrade pip` after installing
- `tensorflow-gpu ⩾ 1.0 and < 2.0`
    - recommend version `1.15` for compatibility with `libcublas.so` and `CUDA`
- `numpy 1.16.6`
- `networkx 1.11`
- `scikit-learn 0.20.4`
- `scipy 1.2.2`
- `matplotlib 2.1.1`

##  required code modifications
clone `https://github.com/daniel-gonzalez-cedre/gae.git` instead of the standard github repo

##  training the model
- run `python train.py`
- run `python train.py --dataset <mydata>`
