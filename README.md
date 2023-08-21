# FC-SIN

The repo is the official code implementation of FC-SIN

## Training
Download the dataset and use the following scripts.
```
python main.py --data_root /path/to/atd12k_points --batch_size 4 --test_batch_size 4 --loss 0.7*L1+0.3*LPIPS
```

## Dataset
Our dataset can be downloaded in the following link.
[Link](https://pan.baidu.com/s/17x0aBshLbM0OXqe0SxHvog)  
Code: ufqn

## Reference
Some great video interpolation resources that we benefit from:

- AnimeInterp [Code](https://github.com/lisiyao21/AnimeInterp.git)
- VFI-Transformer [Code](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer.git)