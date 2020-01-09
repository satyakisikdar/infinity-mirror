#   notes on NetGAN
- to enable GPU, use the `--nv` singularity flag
- remember to make sure that `gpu_id` is always set to a sensible value
    - e.g., don't have `gpu_id=3` if there is only one GPU on the system configured with CUDA
    - if there is only one GPU configured, set `gpu_id=0` wherever encountered

##  requirements
### conda
- `python 3.6`
```singularity
    bootstrap: docker
    from: continuumio/miniconda3:4.4.10
```
- `tensorflow-gpu version 1.4.1`
    - must use `Miniconda 4.4.10`; most current version of Anaconda/Miniconda doesn't work
    - must install `tensorflow-gpu=1.4.1` and NOT `tensorflow=1.4.1`
- `numpy version 1.16.1`
    - must use older version because `allow_pickle=False` throws an exception with `np.load(...)`
    - alternatively, use `np.load(..., allow_pickle=True)` instead of `np.load(...)`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `python-igraph`
- `networkx`
- `numba`
### pip
- `powerlaw`
- `notebook`

## required code modifications
- in file `netgan.py` in directory `netgan`:
    - line 621: 
        - `- y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)`
        + `+ y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)`

## constructing the model
- demo parameters for the `NetGAN(...)` model
    - `gpu_id=0`
    - `use_gumbel=True`
    - `disc_iters=3`
    - `W_down_discriminator_size=128`
    - `W_down_generator_size=128`
    - `l2_penalty_generator=1e-7`
    - `l2_penalty_discriminator=5e-5`
    - `generator_layers=[40]`
    - `discriminator_layers=[30]`
    - `temp_start=5`
    - `learning_rate=0.0003`

##  training the model
### without GPU
    - approximately 4.5 hours to run 2000 iterations of training without GPU
    - for some reason, doesn't seem to be able to finish
- demo parameters for netgan.train() on cell 12 of the notebook:
    - `eval_every = 2000`
    - `plot_every = 2000`
    - `max_patience = 20`
    - `max_iters = 200000`
