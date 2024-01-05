 # CMSC848F Assignment 4


## Data Preparation
Download zip file (~2GB) from https://drive.google.com/file/d/1wXOgwM_rrEYJfelzuuCkRfMmR0J7vLq_/view?usp=sharing. Put the unzipped `data` folder under root directory. There are two folders (`cls` and `seg`) corresponding to two tasks, each of which contains `.npy` files for training and testing.

## Q1. Classification Model (40 points)
Implement the classification model in `models.py`.


 Run `python train.py --task cls` to train the model.Train for 20 epochs minimum.

 For evaluation, run  `python eva_cls.py  --load_checkpoint best_model --batch_size 32` for evaluation. If you want to save the predictions for success and failure cases, the code has to be uncommented and commented out respectively at lines  93 and 114.
 Pass `--num_points ` or `--add_noise True --std ` if you want to check robustness of the model for noise or number of points per object.Accordingly the file paths have to be updated at lines 120/121.
 All outputs are stored under outputs/cls.



## Q2. Segmentation Model (40 points)
 Run `python train.py --task seg` to train the model.Train for 20 epochs minimum.

 For evaluation, run  `python eva_seg.py  --load_checkpoint best_model --batch_size 32 ` for evaluation. If you want to save the predictions for success and failure cases for certain threshold, the code has to be uncommented and commented out respectively at lines  96 and 120. The threshold value has to be updated at lines 109/110 for saving gifs as only failures or success are saved at a time.

 Pass `--num_points ` or `--add_noise True --std ` if you want to check robustness of the model for noise or number of points per object.Accordingly the file paths have to be updated at lines 131/132 and 135/136.
 All outputs are stored under outputs/seg.

## Q3. Robustness Analysis (20 points)
The instructions to check noise or effects of number of points sampled per object are described in the above sections for segmentation and classification repectively.ALl outputs are stored under outputs/seg or outputs/cls.
