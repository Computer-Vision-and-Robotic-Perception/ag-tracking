from .loftr_config import loftr_cfg

# Directore of the data stored
datadir = 'data/LettuceMOT/'
outdir = 'output/LettuceMOT/'

# All possible datasets (8)
sets = ['B&F1', 'B&F2', 'O&I1', 'O&I2', 'straight1', 'straight2', 'straight3', 'straight4']
# sets = ['straight4', 'O&I2', 'O&I1', 'straight1', 'straight3', 'B&F2', 'B&F1', 'straight2']
# The ones to keep
keep = [7]
sets = [seti for i, seti in enumerate(sets) if i in keep]

mode = 'play' # 'play'  'train'  'eval'  'track'  'slam'
tracker = 'agt' # 'clean' 'orig' 'ag'  'agt'
device = 0 #  'cpu' cuda:int
batch_size = 16

## Detector Train
load_checkpoint = 0
epochs = 200
num_workers = 2
backbone = 'ResNet50FPN' # 'ResNet50FPN' 'ResNet101FPN'
# optimizer
lr = 1e-3
momentum = 0.9
weight_decay = 1e-4
# scheduler
step_size = 3
gamma = 0.8

## Detector models (folder) to evaluate
# det_eval_dir = outdir + 'det/straight2-straight4/%s_weights' % backbone
det_eval_dir = outdir + 'det/all/%s_weights' % backbone
det_eval_model = '/model_epoch_201.pt'

loftr_window = 5         # default: 5
loftr_c_threshold = 0.2  # default: 0.2
loftr_border_rm = 2      # default: 2

loftr_cfg['fine_window_size'] = loftr_window
loftr_cfg['match_coarse']['thr'] = loftr_c_threshold
loftr_cfg['match_coarse']['border_rm'] = loftr_border_rm

cfg = {
    'tracker': tracker,
    'loftr_cfg': loftr_cfg,
    'datadir': datadir,
    'outdir': outdir,
    'sets': sets,
    'mode': mode,
    'device': device,
    'batch_size': batch_size,
    'load_checkpoint': load_checkpoint,
    'epochs': epochs,
    'num_workers': num_workers,
    'backbone': backbone,
    'lr': lr,
    'momentum': momentum,
    'weight_decay': weight_decay,
    'step_size': step_size,
    'gamma': gamma,
    'det_eval_dir': det_eval_dir,
    'det_eval_model': det_eval_model,
}
