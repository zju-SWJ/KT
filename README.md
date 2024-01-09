#  Knowledge Translation: Fueling Progress in Model Compression



## Environment

Python 3.9.16, torch 2.0.1



## Purpose of each code file

**construct_train.py:** Generate training dataset

**construct_val.py:** Generate validation dataset

**generate_sh.py:** Generate .sh file to run multiply **construct_train.py** files together

**loader.py:** Dataloader for train.py

**mixer.py:** Translation model for train.py

**net.py:** Models for dataset generation and knowledge translation

**train.py:** Training code for knowledge translation

**utils.py:** Setup for logging



## Generate training and validation datasets

1. Modify **generate_sh.py** and run it. It works fine when running 5 processes on a single 2080Ti gpu (~11GB).

2. Run `nohup sh run.sh &` to generate training dataset.

3. Similiarly, generate validation dataset using the **construct_val.py** file.



## Train the knowledge translation model

The following code can be run on a single A800 gpu (~80GB) and cost roughly a day. You may adjust the **batch_size** and **lr** to run on other gpus.

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
TBD
```