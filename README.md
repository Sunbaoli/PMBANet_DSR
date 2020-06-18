# PMBANet_DSR
# PMBANet: Progressive Multi-Branch Aggregation Network for Scene Depth Super-Resolution

This repo implements the training and testing of depth upsampling networks for "PMBANet: Progressive Multi-Branch Aggregation Network for Scene Depth Super-Resolution" by Xinchen Ye, Baoli Sun, and et al. at DLUT.

## The proposed progressive multi-branch aggregation depth super-resolution framework
![](https://github.com/Sunbaoli/PMBANet_DSR/blob/master/mainnet.png)

## Results
![](https://github.com/Sunbaoli/PMBANet_DSR/blob/master/result1.png)
![](https://github.com/Sunbaoli/PMBANet_DSR/blob/master/result2.png)


This repo can be used for training and testing of depth upsampling under noiseless and noisy cases for Middleburry  datasets. Some trained models are given to facilitate simple testings before getting to know the code in detail. Besides,  the results of our recovered depth maps under both noiseless and noisy cases are all given to make it  easy to compare with and reference our work.

## Dependences

The code supports Python 3

PyTorch (>= 1.1.0)

## Download pretrained models

Download the pretrained model from the Baidu netdisk folder. Link: https://pan.baidu.com/s/1TW9qat987k12iToRYJwPZQ, pwd: z016.
## Train
` pthon train.py `

## Test
` python test.py `

## Citation 
If you find this code useful, please cite:

` Xinchen Ye* et al., PMBANet: Progressive Multi-Branch Aggregation Network for Scene Depth Super-Resolution, Submitted to IEEE Trans. Image Processing, Major revision. `


