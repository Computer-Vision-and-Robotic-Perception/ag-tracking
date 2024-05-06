from .obj_config import AttrDict

def get_config():
    cfg = AttrDict()
    # FRCNN score threshold to keep (new) detections when filtering
    cfg.DETECTION_PERSON_THRESH = 0.5
    # FRCNN score threshold for keeping the track alive (BH: sigma active)
    cfg.REGRESSION_PERSON_THRESH = 0.5
    # NMS threshold 
    cfg.NMS_THRESH = 0.2
    # reduction factor for LoFTR
    cfg.REDUCTION_FACTOR = 3
    # ransac threshold for homography
    cfg.RANSAC_THRES = 3
    return cfg
