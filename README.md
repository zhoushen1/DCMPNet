# Depth Information Assisted Collaborative Mutual Promotion Network for Single Image Dehazing

## Pipeline

![framework](/figs/1.jpg)


## Installation
1. Clone the repository.
    ```bash
    https://github.com/zhoushen1/DCMPNet
    ```

2. Install PyTorch 1.12.0 and torchvision 0.13.0.
    ```bash
    conda install -c pytorch pytorch torchvision
    ```

3. Install the other dependencies.
    ```bash
    pip install -r environment.yaml
    ```
    
## Prepare
Download the RESIDE datasets from https://sites.google.com/view/reside-dehaze-datasets

You need to put the `depth` into the file and you can download the `depth` from https://pan.baidu.com/s/1sNoMlcehMUtSLRuRvsjKKw?pwd=dbcw 
code：dbcw

The final file path should be the same as the following (please check it carefully):
```
┬─ save_models
│   ├─ indoor
│   │   ├─ DIACMPN-dehaze-Indoor.pth
│   │   ├─ DIACMPN-depth-Indoor.pth
│   │   └─ ... (model name)
│   └─ ... (exp name)
└─ data
    ├─ RESIDE-IN
    │   ├─ train
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │       └─ ... (image filename)
    │   └─ test
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │       └─ ... (image filename)
    └─ ... (dataset name)
```

## Training

To customize the training settings for each experiment, navigate to the `configs` folder. Modify the configurations as needed.

After adjusting the settings, use the following script to initiate the training of the model:

```sh
CUDA_VISIBLE_DEVICES=X python train.py --model (model name) --dataset (dataset name) --exp (exp name)
```

For example, we train the DIACMPN-dehaze-Indoor on the ITS:

```sh
CUDA_VISIBLE_DEVICES=0 python train.py --model DIACMPN-dehaze-Indoor --dataset RESIDE-IN --exp indoor
```

## Evaluation

Run the following script to evaluate the trained model with a single GPU.

The initial code and weights are provided https://drive.google.com/drive/folders/1orgiPcrejOI3IChIOCBpw9SzYPuoT9vK?usp=drive_link

```sh
CUDA_VISIBLE_DEVICES=X python test.py --model (model name) --dataset (dataset name) --exp (exp name)
```

For example, we test the DIACMPN-dehaze-Indoor on the SOTS indoor set:

```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model DIACMPN-dehaze-Indoor --dataset RESIDE-IN --exp indoor
```


# Contact:
    Zhou Shen
    School of Computer Science and Engineering, Southeast University                                                        
    Email: zhoushennn@163.com
