import os, sys
import argparse

import numpy as np
import cv2
from skimage import filters

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from linefiller.thinning import thinning
from linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, \
    show_fill_map, my_merge_fill

# for super pixelpooling
from torch_scatter import scatter_mean
from torch_scatter import scatter_add

import softsplat
from forward_warp2 import ForwardWarp
from my_models import create_VGGFeatNet
from vis_flow import flow_to_color


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_root')
    parser.add_argument('output_root')

    parser.add_argument('--label_root', default=None, help='root for label maps')
    parser.add_argument('--start_idx', default=0,
                        help='threshold to differ motion regions from static')
    parser.add_argument('--end_idx', default=None,
                        help='threshold to differ motion regions from static')

    parser.add_argument('--rank_sum_thr', default=0,
                        help='threshold for rank sum')
    parser.add_argument('--height', default=960,
                    help='height of the generated flow, default: 960')
    parser.add_argument('--width', default=540,
                    help='width of the generated flow, default: 540')

    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    ######
    folder_root = args.input_root
    save_root = args.output_root

    label_root = args.label_root

    start_idx = int(args.start_idx)
    end_idx = None if args.end_idx is None else int(args.end_idx)
    use_gpu = args.use_gpu

    # tar_size = (1280, 720)
    tar_size = (args.height, args.width)
    # tar_size = (640, 360)

    rankSumThr = int(args.rank_sum_thr)
    ######

    print('use label maps from %s'%label_root)
    print('use gpu: ', use_gpu)
    sys.stdout.flush()

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    ## make models
    vggNet = create_VGGFeatNet()
    if use_gpu:
        vggNet = vggNet.cuda()

    toTensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])


    totalMatchCount = 0
    folderList = sorted(os.listdir(folder_root))
    if end_idx is None:
        end_idx = len(folderList)

    for f_idx, folder in enumerate(folderList[start_idx:end_idx]):
        f_idx += start_idx
        # if f_idx > 1 + start_idx:
        #     break

        input_subfolder = os.path.join(folder_root, folder)
        imgFileNames = sorted(os.listdir(input_subfolder))
        temp = []
        for name in imgFileNames:
            if name.__contains__('frame'):
                temp.append(name)
        imgFileNames = temp

        print('-- [%d/%d] %s'%(f_idx, end_idx-1, folder))
        print(imgFileNames)
        sys.stdout.flush()

        img1 = cv2.imread(os.path.join(input_subfolder, imgFileNames[0]))
        img3 = cv2.imread(os.path.join(input_subfolder, imgFileNames[-1]))

        # segmentation
        img1_rs = cv2.resize(img1, tar_size)
        img3_rs = cv2.resize(img3, tar_size)

        if label_root is None:
            if 'Japan' in folder:
                boundImg1 = dline_of(img1_rs, 2, 20, [10,10,10]).astype(np.uint8)
                boundImg3 = dline_of(img3_rs, 2, 20, [10,10,10]).astype(np.uint8)
            else:
                boundImg1 = dline_of(img1_rs, 1, 20, [30,40,30]).astype(np.uint8)
                boundImg3 = dline_of(img3_rs, 1, 20, [30,40,30]).astype(np.uint8)

            ret, binMap1 = cv2.threshold(boundImg1, 220, 255, cv2.THRESH_BINARY)
            ret, binMap3 = cv2.threshold(boundImg3, 220, 255, cv2.THRESH_BINARY)

            print('- trapped_ball_processed()')
            sys.stdout.flush()
            fillMap1 = trapped_ball_processed(binMap1, img1_rs)
            fillMap3 = trapped_ball_processed(binMap3, img3_rs)

            labelMap1 = squeeze_label_map(fillMap1)
            labelMap3 = squeeze_label_map(fillMap3)
        else:
            print('- load labelmap')
            sys.stdout.flush()

            print(os.path.join(label_root, folder, 'labelmap_1.npy'))
            print(os.path.join(label_root, folder, 'labelmap_3.npy'))

            labelMap1 = np.load(os.path.join(label_root, folder, 'labelmap_1.npy'))
            print(labelMap1.shape)
            labelMap3 = np.load(os.path.join(label_root, folder, 'labelmap_3.npy'))
            print(labelMap3.shape)
