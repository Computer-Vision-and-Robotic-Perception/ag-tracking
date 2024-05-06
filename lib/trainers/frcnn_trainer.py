import os
import sys
import cv2
import tqdm
import torch
import shutil
import datetime
import numpy as np
import matplotlib.pyplot as plt
from ..utils.det_utils import utils
from torch.utils.data import DataLoader
from ..models.frcnn_fpn import FRCNN_FPN
from ..utils.det_utils.engine import train_one_epoch, evaluate

class FrcnnTrainer():
    def __init__(self, cfg, dset):
        self.cfg = cfg
        # Check and create the save path
        if cfg['mode'] == 'train':
            path = cfg['outdir'] + 'det/frcnn/train/'
            path += datetime.datetime.now().strftime("%y.%m.%d.%H.%M")
            shutil.copytree('config', path + '/config')
            self.save_path = path
            self.weights_path = path + '/%s_weights' % cfg['backbone']
        self.dataset = dset
        self.epoch_start = cfg['epoch_start']
        self.num_epochs = cfg['epochs']
        self.loader = DataLoader(dset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], collate_fn=utils.collate_fn)
        self.model = FRCNN_FPN(backbone_type=cfg['backbone'])
        if self.cfg['checkpoint']:
            print('loading checkpoint %d: "%s"' % (cfg['epoch_start'], cfg['checkpoint']))
            model_state_dict = torch.load(cfg['checkpoint'])
            self.model.load_state_dict(model_state_dict)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
        self.device = torch.device('cuda:0')
    
    def train(self):
        os.makedirs(self.weights_path, exist_ok=True)
        sys.stdout = open(self.save_path + '/train.log', 'w')
        sys.stderr = sys.stdout
        self.model.to(self.device) 
        for epoch in range(self.epoch_start, self.num_epochs + 1):
            print('\n++++++++++++++++++++')
            print('>>> Epoch %03d ++++++' % (epoch))
            print('++++++++++++++++++++')
            train_one_epoch(self.model, self.optimizer, self.loader, self.device, epoch, print_freq=10)
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), self.weights_path + "/model_epoch_{}.pt".format(epoch))
                evaluate(self.model, self.loader, self.device) # This evaluates in the training set
            sys.stdout.flush()

    def test(self):
        if self.cfg['checkpoint']:
            print("Using checkpoint:", self.cfg['checkpoint'])
        else:
            print("WARNING: No checkpoint provided. Using random weights.")

        stat_description = ['APa', 'AP.5', 'AP.75', 'APs', 'APm', 'APl',
                            'AR1', 'AR10', 'AR100', 'APs', 'APm', 'APl']
        print(stat_description[0], stat_description[8])
        
        self.model.to(self.device)
        evaluator = evaluate(self.model, self.loader, self.device)
        stats = list(evaluator.coco_eval.values())[0].stats
        for desc, stat in zip(stat_description, stats):
            print(desc, stat, '*' if desc in ['APa', 'AR100'] else '')

    def predict(self, img=None):
        # Check and create the save path
        path = self.cfg['outdir'] + 'det/frcnn/predict/'
        path += datetime.datetime.now().strftime("%y.%m.%d.%H.%M")
        os.makedirs(path, exist_ok=True)
        
        self.model.to(self.device)
        self.model.eval()
        if img is not None:
            res = self.model(img)
            print(res.shape)
            # TODO: Implement the visualization of the results
            # Save in det/predict/date/name.jpg
            return
        confs = np.array([])
        with torch.no_grad():
            for i, (imgs, target) in tqdm.tqdm(enumerate(self.loader), desc='Predicting'):
                imgs = imgs[0].to(self.device)
                out = self.model([imgs])[0]
                boxes = out['boxes'].cpu().numpy().astype(int)
                confs = np.concatenate([confs, out['scores'].cpu().numpy()])
                img = target[0]['img'].cpu().numpy()
                B, G, R = img[0, :, :], img[1, :, :], img[2, :, :]
                img = np.stack([B, G, R], 2)
                for bb, sc in zip(boxes, out['scores'].cpu().numpy()):
                    x1, y1, x2, y2 = bb
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 1), 2)
                    cv2.putText(img, '%.2f' % sc, (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 0, 0), 2)
                cv2.imshow('video', img)
                cv2.imwrite(path + '/%03d.jpg' % i, (img * 255).astype(np.uint8))
                cv2.waitKey(1)
                if i == 10: break
        plt.figure()
        plt.hist(confs)
        plt.savefig(path + '/000.conf.hist.jpg')
