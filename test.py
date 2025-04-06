import time

import os
import torch
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image as imwrite

import math
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import lpips
from pytorch_msssim import ssim_matlab as ssim

from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms as transforms
import os
from evaluation.distance_transform_v0 import ChamferDistance2dMetric
import torch

import config
import myutils
from loss import Loss

def load_checkpoint(args, model, optimizer, path):
    print("loading checkpoint %s" % path)
    checkpoint = torch.load(path)
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = checkpoint.get("lr", args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)

save_loc = os.path.join(args.checkpoint_dir, "checkpoints")

device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

if args.dataset == 'std12k':
    from dataset.std12k import get_loader
    train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
else:
    raise NotImplementedError

if args.model == 'SAIN':
    from model.SAIN import SAIN

print("Building model: %s"%args.model)
if args.model == 'SAIN':
    args.device = device
    args.resume_flownet = False
    model = SAIN(args)

model = torch.nn.DataParallel(model).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('the number of network parameters: {}'.format(total_params))

##### Define Loss & Optimizer #####
criterion = Loss(args)

from torch.optim import Adamax
optimizer = Adamax(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

scaler = GradScaler()

def calc_psnr(pred, gt):
    return -10 * math.log10(((pred - gt) * (pred - gt)).mean())


def testt(args, epoch):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()
    criterion.eval()
    torch.cuda.empty_cache()

    t = time.time()
    with torch.no_grad():
        for i, (images, gt_image, datapath, flow) in enumerate(tqdm(test_loader)):
            images = [img_.to(device) for img_ in images]
            points = torch.cat([images[3]], dim=1)
            out = model(images[0], images[1], points, flow)

            gt = gt_image.to(device)
            for idx in range(out.size()[0]):
                os.makedirs(args.result_dir + '/' + datapath[idx])
                imwrite(out[idx], args.result_dir + '/' + datapath[idx] + '/sain.png')

            # Save loss values
            loss, loss_specific = criterion(out, gt)
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            # Evaluate metrics
            myutils.eval_metrics(out, gt, psnrs, ssims)

    return losses['total'].avg, psnrs.avg, ssims.avg


def print_log(epoch, num_epochs, one_epoch_time, oup_pnsr, oup_ssim, Lr):
    print('({0:.0f}s) Epoch [{1}/{2}], Val_PSNR:{3:.2f}, Val_SSIM:{4:.4f}'
          .format(one_epoch_time, epoch, num_epochs, oup_pnsr, oup_ssim))
    # write training log
    with open('./training_log/train_log.txt', 'a') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}, Lr:{6}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    one_epoch_time, epoch, num_epochs, oup_pnsr, oup_ssim, Lr), file=f)

""" Entry Point """
def main(args):
    # load_checkpoint(args, model, optimizer, save_loc+'/model_best.pth')
    # test_loss, psnr_val, ssim_val = testt(args, args.start_epoch)
    # print("psnr :{}, ssim:{}".format(psnr_val, ssim_val))

    i = 0

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

    for root, dirs, files in os.walk(args.result_dir):

        for dir in dirs:
            pin = Image.open('{}/{}/sain.png'.format(args.result_dir,dir)).convert("RGB").resize((384, 192))
            pred1n = transform(pin).unsqueeze(0).float() / 255.
            predn = np.asarray(pin)/ 255.

            # gi = Image.open('{}/test_2k_540p/{}/frame2.jpg'.format( args.data_root,dir)).resize((384, 192))
            gi = Image.open('/home/kuhu6123/eccvoutput2/{}/eccvgt.png'.format(dir)).resize((384, 192))
            gt1 = transform(gi).unsqueeze(0).float() / 255.
            gt = np.asarray(gi)/ 255.

            i += 1

            psnrlistn.append(psnr(predn, gt))
            ielistn.append(math.sqrt(mse(predn, gt)))
            cdlistn.append(cd.calc(pred1n, gt1))
            ssimlistn.append(ssim(pred1n, gt1))

            if i % 500 == 0:
                print(i)

    psnr_avgn = np.average(psnrlistn)
    ie_avgn = np.average(ielistn)
    cd_avgn = np.average(cdlistn)
    ssim_avgn = np.average(ssimlistn)

    print("cdval: {}".format(cd_avgn))
    print("ssim: {}".format(ssim_avgn))
    print("psnr: {}".format(psnr_avgn))
    print("ie: {}".format(ie_avgn))

if __name__ == "__main__":
    main(args)
