


from evaluation.pytorch_v0 import *
import kornia

###################### CANNY ######################

def canny(img, a=100, b=200):
    img = I(img).convert('L')
    return I(cv2.Canny(img.cv2(), a, b))

# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def canny_pis(img, sigma=0.33):
    # compute the median of the single channel pixel intensities
    img = I(img).convert('L').uint8(ch_last=False)
    v = np.median(img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img[0], lower, upper)
    # return the edged image
    return I(edged)

# https://en.wikipedia.org/wiki/Otsu%27s_method
def canny_otsu(img):
    img = I(img).convert('L').uint8(ch_last=False)
    high, _ = cv2.threshold(img[0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    low = 0.5 * high
    return I(cv2.Canny(img[0], low, high))

def xdog(img, t=1.0, epsilon=0.04, phi=100, sigma=3, k=1.6):
    img = I(img).convert('L').uint8(ch_last=False)
    grey = np.asarray(img, dtype=np.float32)
    g0 = scipy.ndimage.gaussian_filter(grey, sigma)
    g1 = scipy.ndimage.gaussian_filter(grey, sigma * k)

    #ans = ((1+p) * g0 - p * g1) / 255
    ans = (g0 - t * g1) / 255
    ans = 1 + np.tanh(phi*(ans-epsilon)) * (ans<epsilon)
    return ans

def dog(img, t=1.0, sigma=1.0, k=1.6, epsilon=0.01, kernel_factor=4, clip=True):
    img = I(img).convert('L').tensor()[None]
    kern0 = max(2*int(sigma*kernel_factor)+1, 3)
    kern1 = max(2*int(sigma*k*kernel_factor)+1, 3)
    g0 = kornia.filters.gaussian_blur2d(
        img, (kern0,kern0), (sigma,sigma), border_type='replicate',
    )
    g1 = kornia.filters.gaussian_blur2d(
        img, (kern1,kern1), (sigma*k,sigma*k), border_type='replicate',
    )
    ans = 0.5 + t*(g1-g0) - epsilon
    ans = ans.clip(0,1) if clip else ans
    return ans[0].numpy()

# input: (bs,rgb(a),h,w) or (bs,1,h,w)
# returns: (bs,1,h,w)
def batch_dog(img, t=1.0, sigma=1.0, k=1.6, epsilon=0.01, kernel_factor=4, clip=True):
    # to grayscale if needed
    bs,ch,h,w = img.shape
    if ch in [3,4]:
        img = kornia.color.rgb_to_grayscale(img[:,:3])
    else:
        assert ch==1

    # calculate dog
    kern0 = max(2*int(sigma*kernel_factor)+1, 3)
    kern1 = max(2*int(sigma*k*kernel_factor)+1, 3)
    g0 = kornia.filters.gaussian_blur2d(
        img, (kern0,kern0), (sigma,sigma), border_type='replicate',
    )
    g1 = kornia.filters.gaussian_blur2d(
        img, (kern1,kern1), (sigma*k,sigma*k), border_type='replicate',
    )
    ans = 0.5 + t*(g1-g0) - epsilon
    ans = ans.clip(0,1) if clip else ans
    return ans


############### LOSSES + METRICS ###############

class LineRatioMetric(torchmetrics.Metric):
    def __init__(
            self, convert_dog=True,
            t=2.0, sigma=1.0, k=1.6, epsilon=0.01, kernel_factor=4, clip=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.convert_dog = convert_dog
        self.dog_params = {
            't': t, 'sigma': sigma, 'k': k, 'epsilon': epsilon,
            'kernel_factor': kernel_factor, 'clip': clip,
        }
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.convert_dog:
            preds = (batch_dog(preds, **self.dog_params)>0.5).float()
            target = (batch_dog(target, **self.dog_params)>0.5).float()
        preds = preds.sum((1,2,3))
        target = target.sum((1,2,3))
        dist = torch.nan_to_num(preds/target, nan=1.0, posinf=1.0, neginf=1.0)
        self.running_sum += dist.sum()
        self.running_count += len(dist)
        return
    def compute(self):
        return self.running_sum.float() / self.running_count

class DoGLoss(nn.Module):
    def __init__(
            self, convert_dog=True, mode='l1',
            t=2.0, sigma=1.0, k=1.6, epsilon=0.01, kernel_factor=4, clip=False,
        ):
        super().__init__()
        assert mode in ['l1', 'l2']
        self.convert_dog = convert_dog
        self.mode = mode
        self.dog_params = {
            't': t, 'sigma': sigma, 'k': k, 'epsilon': epsilon,
            'kernel_factor': kernel_factor, 'clip': clip,
        }
        return
    def forward(self, preds, target):
        if self.convert_dog:
            preds = batch_dog(preds, **self.dog_params)
            target = batch_dog(target, **self.dog_params)
        if self.mode=='l1':
            return (preds-target).abs().mean(dim=(1,2,3))
        elif self.mode=='l2':
            return (preds-target).pow(2).mean(dim=(1,2,3))




