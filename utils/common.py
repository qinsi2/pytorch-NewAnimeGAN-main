import torch
import gc
import os
import torch.nn as nn
import urllib.request
import cv2
from tqdm import tqdm
import numpy as np
HTTP_PREFIXES = [
    'http',
    'data:image/jpeg',
]

SUPPORT_WEIGHTS = {
    'hayao',
    'shinkai',
}

ASSET_HOST = 'https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_image(path):
    """
    Read image from given path
    """

    if any(path.startswith(p) for p in HTTP_PREFIXES):
        urllib.request.urlretrieve(path, "temp.jpg")
        path = "temp.jpg"

    return cv2.imread(path)[: ,: ,::-1]


def save_checkpoint(model, optimizer, epoch, args, posfix=''):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    path = os.path.join(args.checkpoint_dir, f'{model.name}{posfix}.pth')
    torch.save(checkpoint, path)


def load_checkpoint(model, checkpoint_dir, posfix=''):
    path = os.path.join(checkpoint_dir, f'{model.name}{posfix}.pth')
    return load_weight(model, path)


def load_weight(model, weight):
    if weight.lower() in SUPPORT_WEIGHTS:
        weight = _download_weight(weight)

    checkpoint = torch.load(weight,  map_location='cuda:0') if torch.cuda.is_available() else \
        torch.load(weight,  map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    epoch = checkpoint['epoch']
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()

    return epoch


def initialize_weights(net):
    for m in net.modules():
        try:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        except Exception as e:
            # print(f'SKip layer {m}, {e}')
            pass


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class DownloadProgressBar(tqdm):
    '''
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    '''
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_weight(weight):
    '''
    Download weight and save to local file
    '''
    filename = f'generator_{weight.lower()}.pth'
    os.makedirs('.cache', exist_ok=True)
    url = f'{ASSET_HOST}/{filename}'
    save_path = f'.cache/{filename}'

    if os.path.isfile(save_path):
        return save_path

    desc = f'Downloading {url} to {save_path}'
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, save_path, reporthook=t.update_to)

    return save_path

# 将图片分割为人物和背景，返回值的第一个是分割的人物，第二个是分割的背景
def seg_person(seg_model, imgs):
    # 对一个batch中的图片进行分割
    seg_imgs = seg_model(imgs)

    seg_person = torch.tensor([])
    seg_bg = torch.tensor([])
    # 将分割后的每个图片的不同目标提取出来
    for seg_img, img in zip(seg_imgs, imgs):
        # 取出各目标的得分
        scores = list(seg_img['scores'].cpu().detach().numpy())
        pred_t = [scores.index(x) for x in scores if x>0.4] # 取出分数足够高的目标的索引，分数是从低到高排列的。
        # 如果分数都不高，那么认为该图片中没有人物目标
        if len(pred_t) == 0:
            seg_bg = torch.cat([seg_bg, img.cpu().unsqueeze(0)])
            continue
        # 取出最后一个目标的索引
        pred_t = pred_t[-1]
        # 取出目标的mask和label
        masks = (seg_img['masks']>0.5).cpu().detach()[:pred_t+1]
        labels = seg_img['labels'].cpu().detach().numpy()[:pred_t+1]
        # 人物目标对应的mask的索引
        personmasks = torch.tensor(np.argwhere(labels == 1))
        if len(personmasks) == 0:
            seg_bg = torch.cat([seg_bg, img.cpu().unsqueeze(0)])
            continue
        object_ = torch.zeros((1, imgs.shape[2], imgs.shape[3]))
        for personmask in personmasks:           
            object_ += masks[personmask].squeeze(0)
            object_ = torch.clip(object_, 0, 1)#.unsqueeze(0)
        object_1 = torch.where(object_ == 0, object_, img.cpu()).unsqueeze(0)
        background = torch.where(object_ == 0, img.cpu(), object_).unsqueeze(0)
        seg_person = torch.cat((seg_person, object_1))
        seg_bg = torch.cat((seg_bg, background))
        # 将每张图片分割后得到的目标存储下来
        # seg_result = torch.cat((seg_result, objects))
    
    return seg_person.to(device) ,seg_bg.to(device)

