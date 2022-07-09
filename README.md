# Introduction
This repository is a reproduce code for **Real-world Anomaly Detection in Survelillance Videos** (CVPR 2018).



## Requirements
* Python 3
* CUDA
* numpy
* tqdm
* [PyTorch](http://pytorch.org/) (1.2)
* [torchvision](http://pytorch.org/)  
Recommend: the environment can be established by run

```
conda env create -f environment.yaml
```


## Data preparation
1. Download the [c3d features][github.com/wanboyang/anomaly_feature.pytorch] and change the "dataset_path" to you/path/data
2. Running the clip2segment.py get feature segments(32 segments for one video)


## Training


```
python main.py --dataset_name UCF_Crime --feature_size 4096 --feature_modal rgb --feature_layer fc6
```

```
python main.py --dataset_name shanghaitech --feature_size 4096 --feature_modal rgb --feature_layer fc6
```

```
python main.py --dataset_name UCSDPed2 --feature_size 4096 --feature_modal rgb --feature_layer fc6 --k 32 --sample_size 4
```

```
python main.py --dataset_name ARD2000 --device 1 --feature_size 8192 --feature_modal rgb --feature_layer pool5 --feature_pretrain_model c3d
```
The models and testing results will be created on ./ckpt and ./results respectively

## Acknowledgements
Thanks the contribution of [W-TALC](https://github.com/sujoyp/wtalc-pytorch) and awesome PyTorch team.
