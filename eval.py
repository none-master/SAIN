
import os
import math
# from skimage.metrics import structural_similarity as ssim
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
from evaluation.distance_transform_v0 import ChamferDistance2dMetric
import torch

def calc_psnr(pred, gt):
    return -10 * math.log10(((pred - gt) * (pred - gt)).mean())

i=0
# ssims = 0.0
# psnrs = 0.0
# ie = 0.0

loss_fn_alex = lpips.LPIPS(net='alex')
transform = transforms.Compose([
    transforms.PILToTensor()
])
cd = ChamferDistance2dMetric()
torch.set_printoptions(precision=4)

psnrlisto = []
ssimlisto = []
ielisto = []
cdlisto = []
psnrlistn = []
ssimlistn = []
ielistn = []
cdlistn = []

for root,dirs,files in os.walk("result"):

    for dir in dirs:
        pio = Image.open('result/{}/sain.png'.format(dir)).convert("RGB").resize((384, 192))
        pred1o = transform(pio).unsqueeze(0).float()/255.
        predo = np.asarray(pio)

        gi = Image.open('/home/kuhu6123/eccvoutput2/{}/eccvgt.png'.format(dir))
        gt1 = transform(gi.resize((384,192))).unsqueeze(0).float()/255.
        gt = np.asarray(gi.resize((384,192)))

        i += 1

        psnrlisto.append(psnr(predo, gt))
        ielisto.append(math.sqrt(mse(predo, gt)))
        cdlisto.append(cd.calc(pred1o, gt1))
        ssimlisto.append(ssim(pred1o, gt1))

        if i%100 == 0:
            print(i)
# ssims/=i
# psnrs/=i
# ie/=i
# cd.compute()

psnr_avgo = np.average(psnrlisto)
ie_avgo = np.average(ielisto)
cd_avgo = np.average(cdlisto)
ssim_avgo = np.average(ssimlisto)

#
# psnr_p = stats.ttest_rel(psnrlisto, psnrlistn)
# psnrnp = np.array(psnrlisto)
# psnr_vo= np.var(psnrnp)
# psnr_vn= np.var(np.array(psnrlistn))
# psnr_avgo = np.average(psnrnp)
#
# ssim_p = stats.ttest_rel(ssimlisto, ssimlistn)
# ssimnp = np.array(ssimlisto)
# ssim_vo= np.var(ssimnp)
# ssim_vn= np.var(np.array(ssimlistn))
# ssim_avgo = np.average(ssimnp)
#
#
# ie_p = stats.ttest_rel(ielisto, ielistn)
# ienp = np.array(ielisto)
# ie_vo= np.var(ienp)
# ie_vn= np.var(np.array(ielistn))
# ie_avgo = np.average(ienp)
#
# cd_p = stats.ttest_rel(cdlisto, cdlistn)
# cdnp = np.array(cdlisto)
# cd_vo= np.var(cdnp)
# cd_vn= np.var(np.array(cdlistn))
# cd_avgo = np.average(cdnp)

# cdres = cd.compute()
print("cdval: {}".format(cd_avgo))
print("ssim: {}".format(ssim_avgo))
print("psnr: {}".format(psnr_avgo))
print("ie: {}".format(ie_avgo))
