import os
import cv2
import yaml
import time
import tqdm
import torch
import kornia
import numpy as np
import kornia.feature as KF
from ultralytics import YOLO
from .tracktor.slam import SLAM
import matplotlib.pyplot as plt
from ..models.frcnn_fpn import FRCNN_FPN
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

def byte_track(cfg, datasets):
    if cfg['detector'] == 'yolov8': model = YOLO(cfg['yolo_maker'])
    else: raise ValueError('Invalid detector for ByteTrack tracker')
    for dataset in datasets:
        print('\n' * 2 + 'dataset:', dataset.imgdir)
        os.makedirs(cfg['outdir'] + 'trackers/bytetrack', exist_ok=True)
        out_file = cfg['outdir'] + 'trackers/bytetrack/' + dataset.imgdir.split('/')[-3] + '.txt'
        file = open(out_file, 'w')
        for i, path in tqdm.tqdm(enumerate(dataset.files), desc='Tracking'):
            results = model.track(source=path, tracker="config/apples_bytetrack.yaml", verbose=False, persist=True)
            if results[0].boxes.id is None: continue
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                left = box[0] - box[2] / 2
                top = box[1] - box[3] / 2
                file.write('%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n' % (i + 1, track_id, left, top, box[2], box[3], 1, 1, 1, 1))
        file.close()

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
    print('Detector: loading checkpoint: "%s"' % cfg['checkpoint'])
    detector.load_state_dict(torch.load(cfg['checkpoint']))
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

    outfile = open(cfg['outdir'] + 'trackers/%s/%s.txt' % (cfg['tracker'], name), 'w')
    os.makedirs((cfg['outdir'] + 'trackers/%s/render/' % cfg['tracker']), exist_ok=True)
    os.makedirs((cfg['outdir'] + 'trackers/%s/times/' % cfg['tracker']), exist_ok=True)
    with torch.no_grad():
        t0, t1 = time.time(), time.time()
        times = []
        for i, (img, target) in tqdm.tqdm(enumerate(loader), desc='Tracking'):
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
    outfile.close()
    times = np.array(times[2:])
    plt.plot(times)
    plt.title("%3.4f, %3.4f, %3.4f" % (times.min(), times.mean(), times.max()))
    plt.savefig(cfg['outdir'] + 'trackers/%s/times/%s.jpg' % (cfg['tracker'], name))
    plt.clf()
    print('Times:', times.min(), times.mean(), times.max())


def slam(cfg, loader, name):
    os.makedirs(cfg['outdir'] + 'slam/render/', exist_ok=True)
    os.makedirs(cfg['outdir'] + 'slam/times', exist_ok=True)
    agt_cfg = cfg_agt.get_config()
    slam = SLAM(cfg, agt_cfg.REDUCTION_FACTOR, agt_cfg.RANSAC_THRES)
    cam_pos = []
    with torch.no_grad():
        t0, t1 = time.time(), time.time()
        times = []
        tmix, tmiy, tmax, tmay = 0, 0, 0, 0
        for i, (imgT, target) in tqdm.tqdm(enumerate(loader)):
            img = target['img'][0].cpu().numpy()
            B, G, R = img[0, :, :], img[1, :, :], img[2, :, :]
            img = np.stack([B, G, R], 2)
            if i == 0:
                im0 = img
                tmix, tmiy, tmax, tmay = 0, 0, img.shape[1], img.shape[0]
            if i == 301: break # straight4:7:301, B&F1:0:361, O&I2:3:851
            h, w = imgT.shape[-2], imgT.shape[-1]
            slam.run_frame(imgT, h, w, i)            
            H0 = slam.H
            if i%10 == 0: cam_pos.append((H0 @ np.array([[img.shape[1]/2, img.shape[0]/2, 1]]).T).flatten()[:2])
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
            frt = cv2.warpPerspective(img, Haux @ H0, (wt, ht), 
                                        borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_NEAREST, borderValue=(1, 1, 1))
            frt_mask = cv2.warpPerspective(np.ones_like(img), Haux @ H0, (wt, ht), 
                                        borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
            full = np.ones((h0, w0, 3))
            full[pmiy-tmiy: pmiy-tmiy + im0.shape[0], pmix-tmix: pmix-tmix + im0.shape[1]] = im0
            frt = (frt_mask == 0) * (full[cmiy-tmiy: cmiy-tmiy+ht, cmix-tmix:cmix-tmix+wt]) + (frt_mask == 1) * frt
            full[cmiy-tmiy: cmiy-tmiy+ht, cmix-tmix:cmix-tmix+wt] = frt
            im0 = full
            # For saving
            imsv = (im0 * 255).astype(np.uint8)
            hsv, wsv, _ = imsv.shape
            imsv = cv2.resize(imsv, (wsv * 2000 // max(hsv, wsv), hsv * 2000 // max(hsv, wsv)), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(cfg['outdir'] + 'slam/render/%s/%s_%04d.jpg' % (name, name, i), imsv)
            # for timing
            t1 = time.time()
            times.append(t1 - t0)
            t0 = t1
        
        name = name.replace('&', 'n')
        cv2.imwrite(cfg['outdir'] + 'slam/%s.jpg' % name, imsv)

        # plot paths alone (no image) using MATPLOTLIB
        plt.figure(figsize=(6, 1.5)) # (6, 1), (6, 1.5), (6, 3)
        x = np.array(cam_pos)[:, 0]
        y = np.array(cam_pos)[:, 1]
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u/2
        pos_y = y[:-1] + v/2
        plt.plot(y, x, 'b')
        plt.quiver(pos_y, pos_x, v, u, color='b', pivot='mid', width=2)
        # Fix aspect ratio
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Path $\mathbf{x}_t$", fontsize=8)
        plt.xlabel("y [pixels]", fontsize=8)
        plt.ylabel("x [pixels]", fontsize=8)
        # set y limits to at least [-50 550]
        plt.ylim([min(-50, min(x)), max(550, max(x))])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        # plt.tight_layout()
        # save
        # plt.savefig(cfg['outdir'] + 'slam/%s_path.jpg' % name, dpi=300)
        # plt.savefig(cfg['outdir'] + 'slam/%s_path.eps' % name, dpi=300)
                
        times = np.array(times[2:])
        plt.figure()
        plt.plot(times)
        plt.savefig(cfg['outdir'] + 'slam/times/%s.jpg' % name)
        plt.clf()

        # Draw the trajectory on imsv using OPENCV
        imbk = imsv.copy()
        # change the values of cam_pos to be in the range of the image
        hsv, wsv, _ = imsv.shape
        cam_pos = np.array(cam_pos)
        cam_pos[:, 0] = cam_pos[:, 0] - tmix
        cam_pos[:, 1] = cam_pos[:, 1] - tmiy
        cam_pos[:, 0] = cam_pos[:, 0] * wsv / (tmax - tmix)
        cam_pos[:, 1] = cam_pos[:, 1] * hsv / (tmay - tmiy)
        cam_pos = cam_pos.astype(int)
        for i in range(len(cam_pos) - 1):
            cv2.arrowedLine(imsv, tuple(cam_pos[i]), tuple(cam_pos[i+1]), (0, 0, 255), thickness=4, tipLength=0.2)
        # cv2.imwrite(cfg['outdir'] + 'slam/%s_image_path_opencv.jpg' % name, imsv)

        # Draw the trajectory on imsv using MATPLOTLIB
        # imsv = imbk.copy()
        imsv = cv2.rotate(imsv, cv2.ROTATE_90_COUNTERCLOCKWISE)
        imsv = cv2.cvtColor(imsv, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 1.5))  # (6, 1), (6, 1.5), (6, 3)
        plt.imshow(imsv)
        # plt.plot(cam_pos[:, 1], wsv - cam_pos[:, 0], 'k', linewidth=0.8)
        # plot arrows using quiver
        x = cam_pos[:, 1]
        y = wsv - cam_pos[:, 0]
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u/2
        pos_y = y[:-1] + v/2
        norm = np.sqrt(u**2 + v**2)
        # plt.quiver(pos_x, pos_y, u/norm, -v/norm, color='r', width=0.004)
        # Fix aspect ratio
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Path $\mathbf{x}_t$", fontsize=8)
        plt.xlabel("y [pixels]", fontsize=8)
        plt.ylabel("x [pixels]", fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        # save
        plt.savefig(cfg['outdir'] + 'slam/%s_image_path_matplotlib.jpg' % name, dpi=300)
        plt.savefig(cfg['outdir'] + 'slam/%s_image_path_matplotlib.eps' % name, dpi=300)
        plt.clf()
        plt.close()
