#   notes on GraphRNN
- to enable GPU, use the `--nv` singularity flag
- remember to make sure that `gpu_id` is always set to a sensible value
    - e.g., don't have `gpu_id=3` if there is only one GPU on the system configured with CUDA
    - if there is only one GPU configured, set `gpu_id=0` wherever encountered

##  requirements
### conda
- `python 3.6`
    - version `4.4.10` of `Miniconda`
- `pytorch 0.4.0`
    - version `1.3.1` works with the code modification `loss.data[0]` -> `loss.item()`
- `tensorflow-gpu 1.5.0`
    - does not work with version `2.0.0`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `networkx 1.11`
- `pyemd`
### pip
- `tensorboard_logger`
- `community`

##  required code modifications
- in file `train.py` in the base directory:
    - line 513: 
        - `- loss.data[0]`
        + `+ loss.item()`
    - line 516: 
        - `- loss.data[0]`
        + `+ loss.item()`
    - line 518: 
        - `- loss.data[0]`
        + `+ loss.item()`

##  constructing the model

##  training the model
- run `python main.py`
