import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib
from models.utils import *
matplotlib.use('Agg')

## Add superglue model to run this script
from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--resize', type=int, nargs='+', default=[512, 288],
    help='Resize the input image before running inference. If two numbers, '
         'resize to the exact dimensions, if one number, resize the max '
         'dimension, if -1, do not resize')
parser.add_argument(
    '--resize_float', action='store_true',
    help='Resize the image after casting uint8 to float')

parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
    help='SuperGlue weights')
parser.add_argument(
    '--max_keypoints', type=int, default=1024,
    help='Maximum number of keypoints detected by Superpoint'
         ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.0001,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--nms_radius', type=int, default=3,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
    ' (Must be positive)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.01,
    help='SuperGlue match threshold')

parser.add_argument(
    '--viz', action='store_true',
    help='Visualize the matches and dump the plots')
parser.add_argument(
    '--eval', action='store_true',
    help='Perform the evaluation'
         ' (requires ground truth pose and intrinsics)')
parser.add_argument(
    '--fast_viz', action='store_true',
    help='Use faster image visualization with OpenCV instead of Matplotlib')
parser.add_argument(
    '--cache', action='store_true',
    help='Skip the pair if output .npz files are already found')
parser.add_argument(
    '--show_keypoints', action='store_true',
    help='Plot the keypoints in addition to the matches')
parser.add_argument(
    '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
    help='Visualization file extension. Use pdf for highest-quality.')
parser.add_argument(
    '--opencv_display', action='store_true',
    help='Visualize via OpenCV before saving output images')
parser.add_argument(
    '--shuffle', action='store_true',
    help='Shuffle ordering of pairs before processing')
parser.add_argument(
    '--force_cpu', action='store_true',
    help='Force pytorch to run in CPU mode.')

opt = parser.parse_args()
print(opt)

if len(opt.resize) == 2 and opt.resize[1] == -1:
    opt.resize = opt.resize[0:1]
if len(opt.resize) == 2:
    print('Will resize to {}x{} (WxH)'.format(
        opt.resize[0], opt.resize[1]))
elif len(opt.resize) == 1 and opt.resize[0] > 0:
    print('Will resize max dimension to {}'.format(opt.resize[0]))
elif len(opt.resize) == 1:
    print('Will not resize images')
else:
    raise ValueError('Cannot specify more than two integers for --resize')


# Load the SuperPoint and SuperGlue models.
device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
print('Running inference on device \"{}\"'.format(device))
config = {
    'superpoint': {
        'nms_radius': opt.nms_radius,
        'keypoint_threshold': opt.keypoint_threshold,
        'max_keypoints': opt.max_keypoints
    },
    'superglue': {
        'weights': opt.superglue,
        'sinkhorn_iterations': opt.sinkhorn_iterations,
        'match_threshold': opt.match_threshold,
    }
}
matching = Matching(config).eval().to(device)

i=0

for root,dirs,files in os.walk("/path/to/dataset"):

    for file in files:
        if not file.__contains__('1'):
            continue
        image = cv2.imread(root + '/' + file)

        # # Load the image pair.
        image0, inp0, scales0 = read_image(
            root + '/' + file, device, opt.resize, 0, opt.resize_float)

        image1, inp1, scales1 = read_image(
            root + '/' + file.replace('1', '3'), device, opt.resize, 0, opt.resize_float)

        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                root + '/' + file, root + '/' + file.replace('1', '3')))
            exit(1)

        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        out = 255 * np.ones((288, 512, 1), np.uint8)
        mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
        for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
            x = np.round((x0 + x1) / 2).astype(int)
            y = np.round((y0 + y1) / 2).astype(int)
            cv2.circle(out, (x, y), radius=0, color=(0, 0, 0), thickness=1)
        cv2.imwrite(str(root + '/inter12.jpg'), out)

        i += 1

print(i)