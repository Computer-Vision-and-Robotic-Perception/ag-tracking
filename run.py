import os
import tqdm
import argparse
from ultralytics import YOLO
from lib.utils.render_utils import play
from torch.utils.data import DataLoader
from lib.utils.track_eval import track_eval
from lib.models.tracker import byte_track, track, slam
from lib.trainers.frcnn_trainer import FrcnnTrainer
from lib.trainers.yolo_trainer import UltraTrainer
from lib.datasets.ag_mot import AgMOT, MergedAgMOT

from config.lettuce_config import cfg as cfg_lettuce
from config.apples_config import cfg as cfg_apples

if __name__ == '__main__':
    # Parse dataset argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='AppleMOTS', help='Dataset to use')
    args = parser.parse_args()
    benchmark = "AppleMOT"
    if 'lettuce' in args.dataset.lower(): 
        cfg = cfg_lettuce
        benchmark = "LettuceMOT"
    if 'apple' in args.dataset.lower(): 
        cfg = cfg_apples
        cfg['datadir'] += 'MOT/'
        cfg['outdir'] += 'MOT/'
    print("Collection:", benchmark)
 
    # Set configurations
    batch = 1 if cfg['task'] in ['play', 'track', 'slam'] else cfg['batch_size']
    shuffle = False if cfg['task'] in ['play', 'track', 'slam'] else True
    train = cfg['task'] == 'det' and cfg['mode'] == 'train'

    # Load sequences as datasets (From MOT Challenge format to AgMOT)
    print('| task:', cfg['task'], '| mode:', cfg['mode'], "| datasets:", cfg['sets'], "|")
    # Safeguard to continue. Needed to avoid accidental execution of the code and data re-write
    inp = input("continue? (y/N): ")
    if inp.lower() != 'y': exit()
    # Collect datasets and dataloaders
    datasets, dataloaders = [], []
    for i, seti in enumerate(cfg['sets']):
        datasets.append(AgMOT(cfg, seti, train=train))
        dataloaders.append(DataLoader(datasets[i], batch_size=batch, shuffle=shuffle, num_workers=cfg['num_workers']))
    #
    if cfg['task'] == 'play':
        for dataloader, name in zip(dataloaders, cfg['sets']):
            print('\n' * 2 + 'dataset:', name)
            play(cfg, dataloader, name)
    # 
    if cfg['task'] == 'det':
        if cfg['detector'] == 'yolov8' and 'apple' in args.dataset.lower(): 
            cfg['datadir'] = cfg['datadir'].replace('/MOT/', '/COCO/')
            cfg['outdir'] = cfg['outdir'].replace('/MOT/', '/COCO/')
        merged = MergedAgMOT(datasets)
        if cfg['detector'] == 'frcnn': trainer = FrcnnTrainer(cfg, merged)
        elif cfg['detector'] == 'yolov8': trainer = UltraTrainer(cfg)
        else: raise ValueError('Invalid detector')
        if cfg['mode'] == 'train': trainer.train()
        if cfg['mode'] == 'eval': trainer.eval() 
        if cfg['mode'] == 'predict': trainer.predict()
    #
    if cfg['task'] == 'slam':
        for dataloader, name in zip(dataloaders, cfg['sets']):
            print('\n' * 2 + 'dataset:', name)
            os.makedirs(cfg['outdir'] + 'slam', exist_ok=True)
            slam(cfg, dataloader, name)
    #
    if cfg['task'] == 'track':
        if cfg['mode'] == 'eval':
            track_eval(cfg, benchmark)
        elif cfg['mode'] == 'predict':   
            if cfg['tracker'] == 'bytetrack' and 'lettuce' in args.dataset.lower(): 
                raise ValueError('ByteTrack tracker is not yet available for LettuceMOT')
            if cfg['tracker'] == 'bytetrack' and 'apple' in args.dataset.lower(): 
                byte_track(cfg, datasets)
            else:
                for dataloader, name in zip(dataloaders, cfg['sets']):
                    print('\n' * 2 + 'dataset:', name)
                    os.makedirs(cfg['outdir'] + 'trackers/' + cfg['tracker'], exist_ok=True)
                    track(cfg, dataloader, name)
        else:
            raise ValueError('Invalid mode: ' + cfg['mode']+ 'for task: track. Use eval or predict. Training is not yet available.')
