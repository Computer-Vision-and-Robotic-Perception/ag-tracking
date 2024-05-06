from .loftr_config import loftr_cfg

# Directory of the data stored
datadir = 'data/AppleMOTS/'
outdir = 'output/AppleMOTS/'

# All possible datasets (12)
sets = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0010', '0011', '0012']
train_set = [seti for i, seti in enumerate(sets) if i in range(6)]
test_set = [seti for i, seti in enumerate(sets) if i in range(6, 12)]
sets = [seti for i, seti in enumerate(sets) if i in range(12)]

## General configs
task = 'det'   # 'play' 'det' 'slam' 'track'
mode = 'test' # 'train' 'test' 'predict'
dectector = 'frcnn' # 'frcnn' 'yolov8'
# TODO: change to the tracker names as defined in the paper
tracker = 'agt' # 'clean' 'orig' 'ag'  'agt' 
device = 0 #  'cpu' cuda:int

## Det Train YOLO Configs
yolo_maker = "output/AppleMOTS/COCO/detect/train/weights/best.pt" # "output/AppleMOTS/COCO/detect/train/weights/best.pt" # "yolov8n.pt"
yolo_data = "config/apple_coco.yaml"

## Det Train FasterRCNN Configs
anchors = 200
num_workers = 8
epoch_start = 0
checkpoint = "output/AppleMOTS/MOT/det/frcnn/train/24.04.15.12.00/ResNet50FPN_weights/model_epoch_200.pt"
epochs = 10
batch_size = 16
backbone = 'ResNet50FPN' # 'ResNet50FPN' 'ResNet101FPN'
# optimizer
lr = 1e-3
momentum = 0.9
weight_decay = 1e-5
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
}
