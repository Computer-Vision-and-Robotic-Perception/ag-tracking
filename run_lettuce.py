from config.lettuce_config import cfg 
from lib.utils.render_utils import play
from torch.utils.data import DataLoader
from lib.models.tracker import track, slam
from lib.utils.data_utils import clean_path
from lib.trainers.det_trainer import DetTrainer
from lib.datasets.lettuce import Lettuce, MergedLettuce

batch = 1 if cfg['mode'] in ['play', 'track', 'slam'] else cfg['batch_size']
shuffle = False if cfg['mode'] in ['play', 'track', 'slam'] else True

if __name__ == '__main__':
    print("datasets:", cfg['sets'], '| mode:', cfg['mode'])
    datasets, dataloaders = [], []
    for i, seti in enumerate(cfg['sets']):
        datasets.append(Lettuce(cfg, seti))
        dataloaders.append(DataLoader(datasets[i], batch_size=batch, shuffle=shuffle))
    
    if cfg['mode'] == 'play':
        for dataloader, name in zip(dataloaders, cfg['sets']):
            print('\n' * 2 + 'dataset:', name)
            clean_path(cfg['outdir'] + 'play')
            clean_path(cfg['outdir'] + 'play/render/%s' % name)
            play(cfg, dataloader, name)
    
    if cfg['mode'] == 'train':
        merged = MergedLettuce(datasets)
        loader = DataLoader(merged, batch_size=batch, shuffle=shuffle)
        if len(datasets) == 8: name = 'all'
        else: name = '-'.join(cfg['sets'])
        trainer = DetTrainer(cfg, merged, name)
        evaluator = trainer.train()

    if cfg['mode'] == 'eval':
        merged = MergedLettuce(datasets)
        loader = DataLoader(merged, batch_size=batch, shuffle=shuffle)
        if len(datasets) == 8: name = 'all'
        else: name = '-'.join(cfg['sets'])
        trainer = DetTrainer(cfg, merged, name)
        evaluator = trainer.eval(cfg['det_eval_dir'])

    if cfg['mode'] == 'track':
        for dataloader, name in zip(dataloaders, cfg['sets']):
            print('\n' * 2 + 'dataset:', name)
            clean_path(cfg['outdir'] + 'track')
            track(cfg, dataloader, name)
        
    if cfg['mode'] == 'slam':
        for dataloader, name in zip(dataloaders, cfg['sets']):
            print('\n' * 2 + 'dataset:', name)
            clean_path(cfg['outdir'] + 'slam')
            slam(cfg, dataloader, name)
 