import cv2
import torch
import kornia
import numpy as np
import kornia.feature as KF
from ...utils.geom_utils import estimate_euclidean_transform

class SLAM():
    def __init__(self, cfg, factor, ransac_t):
        self.device = torch.device(cfg['device'])
        self.matcher = KF.LoFTR(pretrained='outdoor', config=cfg['loftr_cfg']).to(self.device)
        self.matcher.eval()
        self.factor = factor
        self.ransac = ransac_t
        self.imgTpre = None
        self.H = np.eye(3)
    
    def run_frame(self, img, h, w, t, out=torch.tensor(0).cuda()):
        imgT = kornia.color.rgb_to_grayscale(img.to(self.device))
        imgT = kornia.geometry.transform.resize(imgT, [h // self.factor, w // self.factor])
        if t:
            imgdict = {'image0': self.imgTpre, 'image1': imgT}
            correspondences = self.matcher(imgdict)
            # print('Frame %d, Correspondences:' % i, len(correspondences["keypoints0"]))
            mkpts0 = correspondences['keypoints0'].cpu().numpy() * np.array([self.factor, self.factor])
            mkpts1 = correspondences['keypoints1'].cpu().numpy() * np.array([self.factor, self.factor])
            # get the Euclidean transform
            H = np.eye(3)
            H[:2, :], _ = cv2.estimateAffinePartial2D(mkpts1, mkpts0, method=cv2.RANSAC, ransacReprojThreshold=self.ransac)
            H[:2, :2] = H[:2, :2] / np.sqrt(np.linalg.det(H[:2, :2]))
            self.H = self.H @ H
        self.imgTpre = imgT.detach()


import torch.multiprocessing as mp

class SLAMProcess(mp.Process):
    def __init__(self, cfg, factor, ransac_t):
        super(SLAMProcess, self).__init__()
        self.device = torch.device(cfg['device'])
        self.matcher = KF.LoFTR(pretrained='outdoor', config=cfg['loftr_cfg']).to(self.device)
        self.matcher.eval()
        self.factor = factor
        self.ransac = ransac_t
        self.imgTpre = None
        self.args = None
        self.H = np.eye(3)
    
    def run(self):
        if self.args:
            self.run_frame(*self.args)
        else:
            print('started the SLAM process...')

    def run_frame(self, img, h, w, t):
        imgT = kornia.color.rgb_to_grayscale(img)
        imgT = kornia.geometry.transform.resize(imgT, [h // self.factor, w // self.factor])
        if t:
            imgdict = {'image0': self.imgTpre, 'image1': imgT}
            correspondences = self.matcher(imgdict)
            # print('Frame %d, Correspondences:' % i, len(correspondences["keypoints0"]))
            mkpts0 = correspondences['keypoints0'].cpu().numpy() * np.array([self.factor, self.factor])
            mkpts1 = correspondences['keypoints1'].cpu().numpy() * np.array([self.factor, self.factor])
            # get the Euclidean transform
            H = np.eye(3)
            H[:2, :], _ = cv2.estimateAffinePartial2D(mkpts1, mkpts0, method=cv2.RANSAC, ransacReprojThreshold=self.ransac)
            H[:2, :2] = H[:2, :2] / np.sqrt(np.linalg.det(H[:2, :2]))
            self.H = self.H @ H
        self.imgTpre = imgT.detach()