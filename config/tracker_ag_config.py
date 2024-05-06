from .obj_config import AttrDict

def get_config():
    cfg = AttrDict()
    
    cfg.REID_WEIGHTS = 'output/tracktor/reid/res50-mot17-batch_hard/ResNet_iter_25245.pth'
    cfg.REID_CONFIG = 'output/tracktor/reid/res50-mot17-batch_hard/sacred_config.yaml'    

    cfg.TRACKER = AttrDict()
    # FRCNN score threshold to keep (new) detections when filtering
    cfg.DETECTION_PERSON_THRESH = 0.5
    # FRCNN score threshold for keeping the track alive (BH: sigma active)
    cfg.REGRESSION_PERSON_THRESH = 0.5
    # NMS threshold 
    cfg.NMS_THRESH = 0.2
    # Use siamese network to do reid
    cfg.DO_REID = False
    # How much timesteps dead tracks are kept and cosidered for reid
    cfg.INACTIVE_PATIENCE = 2000  # frames
    # How much last appearance features are to keep
    cfg.MAX_FEATURES_NUM = 5
    # How similar do image and old track need to be to be considered the same person
    cfg.REID_SIM_THRESHOLD = 0.8
    # How much IoU do track and image need to be considered for matching. TODO Quit
    cfg.REID_IOU_THRESHOLD = 0.0  # 0.2 

    return cfg