import os
import math
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import lpips
from sklearn.metrics import mean_squared_error
from pytorch_msssim import ssim_matlab as ssim
import scipy.stats as stats

from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms as transforms
import os
from distance_transform_v0 import ChamferDistance2dMetric
import torch

def calc_psnr(pred, gt):
    return -10 * math.log10(((pred - gt) * (pred - gt)).mean())

i=0

loss_fn_alex = lpips.LPIPS(net='alex')
transform = transforms.Compose([
    transforms.PILToTensor()
])
cd = ChamferDistance2dMetric()
torch.set_printoptions(precision=4)

psnrlistn = []
ssimlistn = []
ielistn = []
cdlistn = []

for root,dirs,files in os.walk("/path/to/result"):

    for dir in dirs:

        pin = Image.open('path-toresult/{}/filename.png'.format(dir)).convert("RGB").resize((384,192))
        pred1n = transform(pin).unsqueeze(0).float()/255.
        predn = np.asarray(pin)/255.

        gi = Image.open('/path-toGT/{}/filename.png'.format(dir)).convert("RGB").resize((384,192))
        gt1 = transform(gi).unsqueeze(0).float()/255.
        gt = np.asarray(gi)/255.

        i += 1

        psnrlistn.append(psnr(predn, gt))
        ielistn.append(math.sqrt(mse(predn, gt)))
        cdlistn.append(cd.calc(pred1n, gt1))
        ssimlistn.append(ssim(pred1n, gt1))

psnr_avgn = np.average(psnrlistn)
ie_avgn = np.average(ielistn)
cd_avgn = np.average(cdlistn)
ssim_avgn = np.average(ssimlistn)

print("cdval: {}".format(cd_avgn))
print("ssim: {}".format(ssim_avgn))
print("psnr: {}".format(psnr_avgn))
print("ie: {}".format(ie_avgn))
