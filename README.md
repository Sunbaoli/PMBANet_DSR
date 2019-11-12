# PMBANet_DSR
# PMBANet: Progressive Multi-Branch Aggregation Network for Scene Depth Super-Resolution

This repo implements the training and testing of depth upsampling networks for "PMBANet: Progressive Multi-Branch Aggregation Network for Scene Depth Super-Resolution" by Xinchen Ye, Baoli Sun, and et al. at DLUT.

## The proposed progressive multi-branch aggregation depth super-resolution framework
![](https://github.com/Sunbaoli/DSR/blob/master/code/fig2.png)

## Results
![](https://github.com/Sunbaoli/DSR/blob/master/code/fig1.png)


This repo can be used for training and testing of depth upsampling under noiseless and noisy cases for Middleburry  datasets. Some trained models are given to facilitate simple testings before getting to know the code in detail. Besides,  the results of our inferred edge maps, recovered depth maps under both noiseless and noisy cases are all given to make it  easy to compare with and reference our work.

## Dependences

matlab r2017a

matconvnet-1.0-beta25

## Train
` run start_train.m `

Training on Middlebury noisy depth maps, you can use the following code to preproccess training data:

` im_depth=imnoise(im_depth,'gaussian',0,(5/255)^2);noisy_depth=im_depth; `

## Test
` run test_classSR.m `
## Citation 
If you find this code useful, please cite:

` Xinchen Ye* et al., PMBANet: Progressive Multi-Branch Aggregation Network for Scene Depth Super-Resolution, Submitted to Pattern Recognition, Major revision. `


