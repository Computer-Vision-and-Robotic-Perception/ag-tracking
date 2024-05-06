import os
import cv2
import tqdm
import numpy as np


def play(cfg, dataset, name):
    os.makedirs(cfg['outdir'] + 'play/image/' + name, exist_ok=True)
    os.makedirs(cfg['outdir'] + 'play/video', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(cfg['outdir'] + 'play/video/%s.mp4' % (name), fourcc, 10, (810, 1080))
    for i, (imgs, target) in tqdm.tqdm(enumerate(dataset), desc='Playing'):
        # print(name, 'frame', i, target['img'].shape, target['boxes'].shape)   
        img = target['img'][0].numpy() 
        ids = target['ids'][0].numpy().astype(int)
        anns = target['boxes'][0].numpy().astype(int)
        B, G, R = img[0, :, :], img[1, :, :], img[2, :, :]
        img = np.stack([B, G, R], 2)
        for id, bb in zip(ids, anns):
            x1, y1, x2, y2 = bb
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 1), 2)
            cv2.putText(img, str(id), (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (1, 0, 0), 2)
        cv2.imshow('video', img)
        video.write((img * 255).astype(np.uint8))
        cv2.imwrite(cfg['outdir'] + 'play/image/%s/%06d.jpg' % (name, i), (img * 255).astype(np.uint8))
        cv2.waitKey(1)
        # if i == 50: break
    print('Saving video...')
    video.release()
    print('Video saved.')
