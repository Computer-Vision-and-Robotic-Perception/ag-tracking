import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from config.lettuce_config import cfg 
from lib.utils.det_utils import utils
from torch.utils.data import DataLoader
from lib.models.frcnn_fpn import FRCNN_FPN
from lib.utils.data_utils import clean_path
from lib.trainers.det_trainer import DetTrainer
from lib.datasets.lettuce import Lettuce, MergedLettuce


batch = 1 if cfg['mode'] == 'play' else cfg['batch_size']
shuffle = False if cfg['mode'] == 'play' else True

if __name__ == '__main__':

    name = 'all'
    checkpoint = 200
    device = torch.device(cfg['device'])
    
    datasets, dataloaders = [], []
    for i, seti in enumerate(cfg['sets']):
        datasets.append(Lettuce(cfg, seti))
        dataloaders.append(DataLoader(datasets[i], batch_size=batch, shuffle=shuffle))

    merged = MergedLettuce(datasets)
    loader = DataLoader(merged, batch_size=1, shuffle=True, num_workers=cfg['num_workers'], collate_fn=utils.collate_fn)

    # It can de det or det_backup
    save_path = cfg['outdir'] + 'det_backup/' + name + '/' + cfg['backbone'] + '_weights'
    model = FRCNN_FPN(backbone_type=cfg['backbone'])
    chkpt = save_path + '/model_epoch_%d.pt' % checkpoint
    print('loading checkpoint: "%s"' % chkpt)
    model_state_dict = torch.load(chkpt)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    confs = np.array([])
    with torch.no_grad():
        for i, (imgs, target) in enumerate(loader):    
            imgs = imgs[0].to(device)
            out = model([imgs])[0]
            boxes = out['boxes'].cpu().numpy().astype(int)
            confs = np.concatenate([confs, out['scores'].cpu().numpy()])
            print(i, len(confs))
            img = target[0]['img'].cpu().numpy()
            B, G, R = img[0, :, :], img[1, :, :], img[2, :, :]
            img = np.stack([B, G, R], 2)
            for bb, sc in zip(boxes, out['scores'].cpu().numpy()):
                x1, y1, x2, y2 = bb
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 1), 2)
                cv2.putText(img, '%.2f' % sc, (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 0, 0), 2)
            cv2.imshow('video', img)
            cv2.imwrite('output/LettuceMOT/det/test/%03d.png' % i, (img * 255).astype(np.uint8))
            cv2.waitKey(1)
            if i == 1000: break
    plt.figure()
    plt.hist(confs)
    plt.savefig('output/LettuceMOT/det/test/confs.png')
