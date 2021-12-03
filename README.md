
## Crop Classification under Varying Cloud Cover with Neural Ordinary Differential Equations

This is a Pytorch implementation of NODE for crop mapping task described in

[Metzger, Nando, Mehmet Ozgur Turkoglu, Stefano D'Aronco, Jan Dirk Wegner, and Konrad Schindler. "Crop classification under varying cloud cover with neural ordinary differential equations." IEEE Transactions on Geoscience and Remote Sensing (2021).](https://arxiv.org/pdf/2012.02542.pdf)


If you find our work useful in your research, please consider citing our paper:

```bash
@article{metzger2021crop,
  title={Crop classification under varying cloud cover with neural ordinary differential equations},
  author={Metzger, Nando and Turkoglu, Mehmet Ozgur and D'Aronco, Stefano and Wegner, Jan Dirk and Schindler, Konrad},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2021},
  publisher={IEEE}
}
```


<img src="https://github.com/nandometzger/ODEcrop/blob/master/assets/seq.png">

In remote sensing and Earth observation we often have data gaps which translates irregular time series not only due to clouds, but also caused by irregular acquisition patterns in space or time, sensor failures and replacements, transmission limits, etc. For instance, we show below examples of Sentinel-2 time series data from TUM dataset (left for winter wheat, right for corn). Observations obscured
by clouds are marked in blue at the bottom. Note their irregular distribution.
<img src="https://github.com/nandometzger/ODEcrop/blob/master/assets/tum.png">

Illustration of the hidden state dynamics for traditional RNNs (left) and for the ODE-RNN model (right).
<img src="https://github.com/nandometzger/ODEcrop/blob/master/assets/rnn_node.png">


## Setup
We use a Conda environment that makes it easy to install all dependencies. Our code has been tested on Ubuntu 20.04 with PyTorch xx and CUDA xx.

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python 3.8.
2. Create the conda environment: ```conda env create -f environment-xxxx.yml```
3. Activate the environment: ```conda activate xxxx```

## Getting Started
