# 10708 Final Project: Video Content Swap Using GAN
Tingfung Lau, Sailun Xu and Xinze Wang. 

Based on PyTorch implementation of [drnet](https://github.com/edenton/drnet-py) by the orignal author.

## Preparing Dataset
Please refer [drnet](https://github.com/edenton/drnet-py) for downloading and preprocessing the data set.

## Custom training
To run with the original setting
```
python train_drnet.py --dataset kth --data_root ../datasets --image_width 128
```

To use AlphaPose as pose encoder
```
python train_drnet.py --dataset kth --data_root ../datasets --image_width 128 --pose --pose_dim 35
```

To use AlphaPose as pose encoder and use GAN loss
```
python train_drnet.py --dataset kth --data_root ../datasets --image_width 128 --pose --pose_dim 35 --swap_loss gan

```

