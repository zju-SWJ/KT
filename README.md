#  Knowledge Translation: A New Pathway for Model Compression

## Paper

Under review. Preprint at [arXiv](https://arxiv.org/abs/2401.05772).



## Environment

Python 3.9.16, torch 2.0.1



## Purpose of each code file

**construct_train.py:** Generate training dataset

**construct_val.py:** Generate validation dataset

**generate_sh.py:** Generate .sh file to run multiply **construct_train.py** files together

**loader.py:** Dataloader for **train.py**

**mixer.py:** Translation model for **train.py**

**net.py:** Models for dataset generation and knowledge translation

**train.py:** Training code for knowledge translation

**utils.py:** Setup for logging



## Generate training and validation datasets (or download [train_data.tar](https://1drv.ms/u/s!AgtEQfmiuIJNgot5Ag58uY7GWcqOFA?e=JI8MIE) and [val_data.tar](https://1drv.ms/u/s!AgtEQfmiuIJNgot4HQxWPmfaXes38g?e=gMzaAJ) from OneDrive)

1. Modify **generate_sh.py** and run it. It works fine when running 5 processes on a single 2080Ti GPU (~11GB).

2. Run `nohup sh run.sh &` to generate training dataset.

3. Similarly, generate validation dataset using the **construct_val.py** file.



## Train the knowledge translation model

The following code can be run on a single A800 GPU (~80GB) and costs roughly a day. You may adjust the **batch_size** and **lr** to run on other GPUs.

```
python train.py --gpu_id 0 \
                --optimizer adam --lr 0.001 --batch_size 4096 \
                --wd 0. --scheduler cos --warmup 0 --grad_clip -1. \
                --num_layers 24 --s_dim 72 --c_dim 128 --hidden_s_dim 256 --hidden_c_dim 512 \
                --mask_target t --prob 0. \
                --noise_target t --noise 0. \
                --dropout 0.1 \
                --num_epochs 1000 --log_step 25 --train_data_length 300000
```



## Citation

If you find this repository useful, please consider citing the following paper:
```
@article{sun2024knowledge,
  title={Knowledge Translation: A New Pathway for Model Compression},
  author={Sun, Wujie and Chen, Defang and Chen, Jiawei and Feng, Yan and Chen, Chun and Wang, Can},
  journal={arXiv preprint arXiv:2401.05772},
  year={2024}
}
```
