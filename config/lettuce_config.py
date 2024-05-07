from .loftr_config import loftr_cfg

# Directory of the data stored
datadir = 'data/LettuceMOT/'
outdir = 'output/LettuceMOT/'

# All possible datasets (8)
sets = ['B&F1', 'B&F2', 'O&I1', 'O&I2', 'straight1', 'straight2', 'straight3', 'straight4']
train_set = [seti for i, seti in enumerate(sets) if i in [4, 6, 7]]
test_set = [seti for i, seti in enumerate(sets) if i in [0, 1, 2, 3, 5]]
sets = [seti for i, seti in enumerate(sets) if i in [4, 5, 6, 7, 0, 1, 2, 3]] # [4, 5, 6, 7, 0, 1, 2, 3]

## General configs
task = 'track'   # 'play' 'det' 'slam' 'track'
mode = 'eval' # 'train' 'eval' 'predict'
dectector = 'frcnn' # 'frcnn' 'yolov8'
# TODO: change to the tracker names as defined in the paper
tracker = 'agt' # 'clean' 'orig' 'ag'  'agt' 'bytetrack'
device = 0 #  'cpu' cuda:int

## Det Train YOLO Configs
yolo_maker = "yolo8n.pt"
yolo_data = "config/lettuce_coco.yaml" # TBD

## Det Train FasterRCNN Configs
anchors = 200
num_workers = 8
epoch_start = 0
checkpoint = "output/LettuceMOT/det/frcnn/train/24.03.15.12.00/ResNet50FPN_weights/model_epoch_200.pt"
epochs = 1
batch_size = 16
backbone = 'ResNet50FPN' # 'ResNet50FPN' 'ResNet101FPN'
# optimizer
lr = 1e-3
momentum = 0.9
weight_decay = 1e-4
# scheduler
step_size = 3
gamma = 0.8

## LOFTR configs (Tracker)
loftr_window = 5         # default: 5
loftr_c_threshold = 0.2  # default: 0.2
loftr_border_rm = 2      # default: 2

# Those are tracker configs
loftr_cfg['fine_window_size'] = loftr_window
loftr_cfg['match_coarse']['thr'] = loftr_c_threshold
loftr_cfg['match_coarse']['border_rm'] = loftr_border_rm

track_metrics_eval = ["MOTA", "HOTA", "DetA", "AssA", "AssRe", "AssPr", "IDF1", "IDR", "IDP"] # ['MOTA', "HOTA", "IDF1"]

cfg = {
    'tracker': tracker,
    'loftr_cfg': loftr_cfg,
    'datadir': datadir,
    'outdir': outdir,
    'train_set': train_set,
    'test_set': test_set,
    'sets': sets,
    'task': task,
    'mode': mode,
    'detector': dectector,
    'yolo_maker': yolo_maker,
    'yolo_data': yolo_data,
    'device': device,
    'batch_size': batch_size,
    "epoch_start": epoch_start,
    'checkpoint': checkpoint,
    'epochs': epochs,
    'num_workers': num_workers,
    'backbone': backbone,
    'lr': lr,
    'momentum': momentum,
    'weight_decay': weight_decay,
    'step_size': step_size,
    'gamma': gamma,
    'track_metrics_eval': track_metrics_eval,
}
