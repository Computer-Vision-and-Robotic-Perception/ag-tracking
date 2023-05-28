import cv2
import yaml
import time
import torch
import kornia
import numpy as np
import kornia.feature as KF
from .tracktor.slam import SLAM
import matplotlib.pyplot as plt
from ..models.frcnn_fpn import FRCNN_FPN
from ..utils.data_utils import clean_path
from .tracktor.reid.resnet import resnet50

# from .tracktor.tracker_clean import Tracker
from .tracktor import tracker_ag as ag
from .tracktor import tracker_agt as agt
from .tracktor import tracker_orig as orig
from .tracktor import tracker_clean as clean

from config import tracker_ag_config as cfg_ag 
from config import tracker_agt_config as cfg_agt 
from config import tracker_orig_config as cfg_orig 
from config import tracker_clean_config as cfg_clean 

np.set_printoptions(precision=3, suppress=True)

def track(cfg, loader, name):

    if cfg['tracker'] == 'clean': tracker_cfg = cfg_clean.get_config()
    if cfg['tracker'] == 'orig': tracker_cfg = cfg_orig.get_config()
    if cfg['tracker'] == 'agt': tracker_cfg = cfg_agt.get_config()
    if cfg['tracker'] == 'ag': tracker_cfg = cfg_ag.get_config()

    # set the device
    device = torch.device(cfg['device'])

    # set the detector
    print('Detector: setting up...')
    detector = FRCNN_FPN(cfg['backbone'])
    chkpt = cfg['det_eval_dir'] + cfg['det_eval_model']
    print('Detector: loading checkpoint: "%s"' % chkpt)
    detector.load_state_dict(torch.load(chkpt))
    detector.to(device)
    detector.eval()
    print('Detector ready!')

    if cfg['tracker'] in ['orig', 'ag']:
        # set the ReID network
        print('ReID network: setting up...')
        with open(tracker_cfg.REID_CONFIG) as file:
            reid_config = yaml.load(file, Loader=yaml.FullLoader)
        reid_network = resnet50(pretrained=False, **reid_config['reid']['cnn'])
        print('ReID network: loading checkpoint: "%s"' % tracker_cfg.REID_WEIGHTS)
        reid_network.load_state_dict(torch.load(tracker_cfg.REID_WEIGHTS))
        reid_network.to(device)
        reid_network.eval()
        print('ReID network ready!')
    
    # set the Tracker
    if cfg['tracker'] == 'clean': tracker = clean.Tracker(detector, tracker_cfg)
    if cfg['tracker'] == 'orig': tracker = orig.Tracker(detector, reid_network, tracker_cfg)
    if cfg['tracker'] == 'agt': tracker = agt.Tracker(detector, cfg, tracker_cfg)
    if cfg['tracker'] == 'ag': tracker = ag.Tracker(detector, reid_network, tracker_cfg)

    outfile = open(cfg['outdir'] + 'track/' + name + '.txt', 'w')
    clean_path(cfg['outdir'] + 'track/render')
    clean_path(cfg['outdir'] + 'track/times')
    with torch.no_grad():
        t0, t1 = time.time(), time.time()
        times = []
        for i, (img, target) in enumerate(loader):
            imgT = img[0][None, :, :, :].to(device)
            img = target['img'][0].cpu().numpy()
            B, G, R = img[0, :, :], img[1, :, :], img[2, :, :]
            img = np.stack([B, G, R], 2)
            dets, features, sizes = detector.detect_batches(imgT)
            tracker.step({'img': imgT, 'net_in_sizes': sizes[0], 'features': features}, dets[0])
            results = tracker.get_results()
            for track in results.keys():
                for frame in results[track].keys():
                    if frame == i:
                        x1, y1, x2, y2, conf = results[track][frame]
                        x1, y1, x2, y2 = int(x1), int (y1), int(x2), int(y2)
                        outfile.write('%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n' % (i + 1, track, x1, y1, x2 - x1, y2 - y1, conf, 1, 1, 1))
            #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 1), 3)
            #             cv2.putText(img, str(track), (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (1, 0, 0), 3)
            # cv2.imshow('result', np.transpose(img, (1, 0, 2)))
            # cv2.imshow('result', img)
            # if i%1 == 0: cv2.imwrite(cfg['outdir'] + 'track/render/%d.jpg' % (i + 1), (img * 255).astype(np.uint8))
            # cv2.waitKey(0)
            t1 = time.time()
            times.append(t1 - t0)
            t0 = t1
            print('frame', i)
    outfile.close()
    times = np.array(times[2:])
    plt.plot(times)
    plt.title("%3.4f, %3.4f, %3.4f" % (times.min(), times.mean(), times.max()))
    plt.savefig(cfg['outdir'] + 'track/times/%s.png' % name)
    plt.clf()
    print('Times:', times.min(), times.mean(), times.max())


def slam(cfg, loader, name):
    clean_path(cfg['outdir'] + 'slam/render/' + name)
    clean_path(cfg['outdir'] + 'slam/times')
    agt_cfg = cfg_agt.get_config()
    slam = SLAM(cfg, agt_cfg.REDUCTION_FACTOR, agt_cfg.RANSAC_THRES)
    with torch.no_grad():
        t0, t1 = time.time(), time.time()
        times = []
        tmix, tmiy, tmax, tmay = 0, 0, 0, 0
        for i, (imgT, target) in enumerate(loader):
            img = target['img'][0].cpu().numpy()
            B, G, R = img[0, :, :], img[1, :, :], img[2, :, :]
            img = np.stack([B, G, R], 2)
            if i == 0:
                im0 = img
                tmix, tmiy, tmax, tmay = 0, 0, img.shape[1], img.shape[0]
            if i == 2000: break
            h, w = imgT.shape[-2], imgT.shape[-1]
            slam.run_frame(imgT, h, w, i)            
            H0 = slam.H
            # Rendering for validation purposes
            corners = np.array([[0, w, 0, w],[0, 0, h, h],[1, 1, 1, 1]])
            warpcorners = H0 @ corners
            wc = np.round(warpcorners[:2, :]/warpcorners[2, :]).astype(int)
            pmix, pmiy, pmax, pmay = tmix, tmiy, tmax, tmay
            cmix, cmiy, cmax, cmay = wc[0, :].min(), wc[1, :].min(), wc[0, :].max(), wc[1, :].max()
            tmix, tmiy, tmax, tmay = min(cmix, tmix), min(cmiy, tmiy), max(cmax, tmax), max(cmay, tmay)
            # modify translation for a better warping
            Haux = np.array([[1.0, 0.0, -cmix], [0.0, 1.0, -cmiy], [0, 0, 1]])
            # get dimensions
            wt, ht = cmax - cmix, cmay - cmiy
            w0, h0 = tmax - tmix, tmay - tmiy
            # warp current frame
            frt = cv2.warpPerspective(img, Haux @ H0, (wt, ht), borderMode=cv2.BORDER_REPLICATE)
            full = np.zeros((h0, w0, 3))
            full[pmiy-tmiy: pmiy-tmiy + im0.shape[0], pmix-tmix: pmix-tmix + im0.shape[1]] = im0
            frt += (frt == 0) * (full[cmiy-tmiy: cmiy-tmiy+ht, cmix-tmix:cmix-tmix+wt])
            full[cmiy-tmiy: cmiy-tmiy+ht, cmix-tmix:cmix-tmix+wt] = frt
            im0 = full
            cv2.imwrite(cfg['outdir'] + 'slam/render/%s/%06d.png' % (name, i), (im0 * 255).astype(np.uint8))
            t1 = time.time()
            times.append(t1 - t0)
            t0 = t1
            print('frame', i)
        cv2.imwrite(cfg['outdir'] + 'slam/%s.png' % name, (im0 * 255).astype(np.uint8))
        times = np.array(times[2:])
        plt.plot(times)
        plt.savefig(cfg['outdir'] + 'slam/times/%s.png' % name)
        plt.clf()
        print('Times:', times.min(), times.mean(), times.max())
