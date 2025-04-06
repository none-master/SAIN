# SAIN

The repo is the code implementation for paper "Bridging the Gap: Sketch-Aware Interpolation Network
for High-Quality Animation Sketch Inbetweening".

[Paper](https://arxiv.org/abs/2308.13273)

## Training
Download the dataset and use the following scripts.
```
python main.py --data_root /path/to/std12k_points --batch_size 4 --test_batch_size 4 --loss 0.7*L1+0.3*LPIPS
```

## Dataset
Our dataset can be downloaded in the following link.
[google drive](https://drive.google.com/file/d/1vyu_ePFN9sFjqxc-sPdSWuSCLnWFVUT7/view?usp=sharing)

The pre-trained model can be downloaded in the following link
[google drive](https://drive.google.com/file/d/1bPvGtm9Ty-ALrHc_NdzJQUes8elOBZfa/view?usp=sharing)

The training & test dataset containing region & stroke level correspondence data can be downloaded in the following link.
[google drive](https://drive.google.com/file/d/1VMr2oPQCqUE579dnY4eFGGVAhrgjVR2V/view?usp=sharing)

## Reference
Some great video interpolation resources that we benefit from:

- AnimeInterp [Code](https://github.com/lisiyao21/AnimeInterp.git)
- VFI-Transformer [Code](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer.git)
- EISAI [Code](https://github.com/ShuhongChen/eisai-anime-interpolator.git) 

## Citation

```
@inproceedings{shen2023sain,
    title={Bridging the Gap: Sketch-Aware Interpolation Network for High-Quality Animation Sketch Inbetweening},
    author={Jiaming Shen, Kun Hu, Wei Bao, Chang Wen Chen, and Zhiyong Wang},
    Booktitle = {Proc. of ACM International Conference on Multimedia (MM’24)},
    year={2024}
}

```
