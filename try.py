from skimage.metrics import peak_signal_noise_ratio as psnr

from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms as transforms
import torch

transform = transforms.Compose([
    transforms.PILToTensor()
])
torch.set_printoptions(precision=4)

dir = 'Disney_v4_0_000024_s2'
pio = Image.open('result/{}/sain.png'.format(dir)).convert("RGB").resize((384, 192))
predo = np.asarray(pio)

gi = Image.open('/home/kuhu6123/eccvoutput2/{}/eccvgt.png'.format(dir))
gt = np.asarray(gi.resize((384,192)))

print(psnr(predo, gt))


