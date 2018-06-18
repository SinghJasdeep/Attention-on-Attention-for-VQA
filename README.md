# [Attention on Attention:  Architectures for Visual Question Answering (VQA)](http://arxiv.org/abs/1803.07724)

This is the code for our paper by the same name. Link in the title. 

This Project was done for Stanford's CS 224N and CS 230.

Our model architecture is inspired by the winning entry of the [2017 VQA Challenge](http://www.visualqa.org/challenge.html).

Which follows the VQA system described in ["Bottom-Up and
Top-Down Attention for Image Captioning and Visual Question Answering"](https://arxiv.org/abs/1707.07998) 
and ["Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge"](https://arxiv.org/abs/1708.02711).

## Licence 
MIT

## Our Architectures 
![Models](https://github.com/SinghJasdeep/Attention-on-Attention-for-VQA/blob/master/images/VQA_Architecture.png)

![Models](https://github.com/SinghJasdeep/Attention-on-Attention-for-VQA/blob/master/images/Parameter_Tree.png)


## This Project uses code provided [here](https://github.com/hengyuan-hu/bottom-up-attention-vqa)
We used the preprocessing and base code provided by the above link and then performed an extensive architecture and hyperparameter search. 

## Results

| Model | Validation Accuracy | Training Time
| --- | --- | -- |
| Reported Model | 63.15 | 12 - 18 hours (Tesla K40) |
| Our A3x2 Model | **64.78** | 4 hours AWS g3.8xlarge instance (2x M60) |

The accuracy was calculated using the [VQA evaluation metric](http://www.visualqa.org/evaluation.html).

## About
This is part of a project done for Stanford's CS 224N and CS 230. 

## Implementation Details

Check out our paper for the full implemetation details and hyperparamter search. 
ArXiv link coming soon. 

## HyperParameters Search 

![Hyper](https://github.com/SinghJasdeep/Attention-on-Attention-for-VQA/blob/master/images/Hyperparams.png)

## Dual Attention Visualization

![HeatMaps](https://github.com/SinghJasdeep/Attention-on-Attention-for-VQA/blob/master/images/Attention.png)

## Usage

#### Prerequisites

Make sure you are on a machine with a NVIDIA GPU and Python 2.7+ with about 70 GB disk space.

1. Install [PyTorch](http://pytorch.org/) with CUDA and Python 2.7.
2. Install [h5py](http://docs.h5py.org/en/latest/build.html).

#### Data Setup

All data should be downloaded to a data/ directory in the root directory of this repository.

The easiest way to download the data is to run the provided script `tools/download.sh` from the repository root. If the script does not work, it should be easy to examine the script and modify the steps outlined in it according to your needs. Then run `tools/process.sh` from the repository root to process the data to the correct format.

#### Training

Simply run `python main.py` to start training. The default model run is the best performing A3x2. Other model variations can be run using the models flag. The training and validation scores will be printed every epoch, and the best model will be saved under the directory "saved_models". The default flags should give you the result provided in the table above.

#### Pre-Trained Models 

Certain Pretrained models availible upon request. 

## [Our Paper](http://arxiv.org/abs/1803.07724)

![Paper](https://github.com/SinghJasdeep/Attention-on-Attention-for-VQA/blob/master/Paper/Attn_on_Attn.pdf)

## [Citation](http://dblp.uni-trier.de/rec/bibtex/journals/corr/abs-1803-07724): 

Please use the Citation found at: 

http://dblp.uni-trier.de/rec/bibtex/journals/corr/abs-1803-07724

