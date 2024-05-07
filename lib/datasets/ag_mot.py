import os
import cv2
import torch
import kornia
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from ..utils.data_utils import get_mot_gt
from ..utils.det_utils.utils import get_transform


class AgMOT(Dataset):
    def __init__(self, cfg, name, train=False, anchors=200):
        self.train = train
        self.transforms = get_transform(self.train)
        self.anchors = anchors
        # Read ground-truth bounding boxes
        self.gts = get_mot_gt(cfg['datadir'] + name + '/gt/gt.txt')
        # Image location and list
        self.imgdir = cfg['datadir'] + name + '/img/'
        self.files = sorted([self.imgdir + img for img in os.listdir(self.imgdir)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Read image
        imgPIL = Image.open(self.files[index]).convert("RGB")
        img = cv2.imread(self.files[index])
        img = kornia.image_to_tensor(img, True).float() / 255.0
        # Get annotations
        anns = [ann for ann in self.gts if ann[0] == index + 1]
        if anns: anns = torch.tensor(anns)
        else:    anns = torch.zeros(0, 6) 
        anntensor = -torch.ones(self.anchors, 6)
        anntensor[:len(anns), :] = anns
        # If train, return only corresponding bounding boxes
        if self.train: anntensor = anntensor[:len(anns), :]
        ids = anntensor[:, 1]
        boxes = anntensor[:, -4:]
        labels = torch.minimum(anntensor[:, 0], torch.tensor(1)).to(dtype = torch.int64)
        image_id = torch.tensor(index)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros(self.anchors, dtype=torch.int64)
        target = {'img': img, 'boxes': boxes, 'ids': ids, 'labels': labels, 
                  'image_id': image_id, 'area': area, 'iscrowd': iscrowd}

        if self.transforms is not None:
            imgPIL, target = self.transforms(imgPIL, target)

        return imgPIL, target

class MergedAgMOT(AgMOT):
    def __init__(self, dsets):
        self.train = dsets[0].train
        self.transforms = get_transform(self.train)
        self.anchors = dsets[0].anchors
        # Initialize with the first dataset
        self.gts = dsets[0].gts 
        self.files = dsets[0].files
        # For the remaining datasets, append:
        for dset in dsets[1:]:        
            npre = len(self.files)
            self.files += dset.files
            # for each frame in the dataset: change image_id
            for trk in dset.gts:
                trk[0] += npre
            self.gts += dset.gts
