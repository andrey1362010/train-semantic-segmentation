import os

import cv2
import numpy as np
import torch
import argparse
import yaml
import math

from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from tqdm import tqdm

from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text


class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette))
        print(self.model)
        checkpoint = torch.load("/home/andrey/IdeaProjects/semantic-segmentation/tools/output/SegFormer_MiT-B1_TRAINS8K.pth")
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]

        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32

        # resize the image
        image = T.Resize((nH, nW))(image)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        # get segmentation map (value being 0 to num_classes)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # # convert segmentation map to color map
        # seg_image = self.palette[seg_map].squeeze()
        # if overlay:
        #     seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)
        #
        # image = draw_text(seg_image, seg_map, self.labels)
        return seg_map

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
        
    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        image = io.read_image(img_fname)
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map


TEST_PATH = "/media/andrey/big/datasets/hacksAI/TaskRZD/test"
SV_PATH = "/media/andrey/big/datasets/hacksAI/TaskRZD/submit"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../configs/trains8k.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    semseg = SemSeg(cfg)

    for name in tqdm(os.listdir(TEST_PATH)):
        img_path = os.path.join(TEST_PATH, name)
        save_path = os.path.join(SV_PATH, name)

        test_file = Path(img_path)

        segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY']).numpy()[0][:, :, np.newaxis].repeat(3, axis=-1)

        res = np.zeros((segmap.shape[0], segmap.shape[1], 3), dtype=np.uint8)
        res[segmap == 1] = 6
        res[segmap == 2] = 7
        res[segmap == 3] = 10

        cv2.imwrite(save_path, res)
