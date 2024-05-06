import os 
import numpy as np
from PIL import Image


def get_mot_gt(file):
    file = open(file, 'r')
    lines = list(file.readlines())
    linesint = []
    for line in lines:
        fields = line[:-1].split(',')
        frame = int(fields[0])
        id = int(fields[1])
        x = int(fields[2])
        y = int(fields[3])
        w = int(fields[4])
        h = int(fields[5])
        linesint.append([frame, id, x, y, x + w, y + h])
    return linesint

## For convertion from AppleMOTS to MOTChallenge and Ultralytics-COCO format
def encode_mask_to_RLE(binary_mask):
    fortran_binary_mask = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(fortran_binary_mask)
    return rle['counts'].decode('ascii') 

def create_mot_n_coco_for_sequence(params):
    print("Processing sequence %s" % params['seq'])
    # MOTS data directories
    mots_images_dir = '%s/%s/images/%s' % (params['input_dir'], params['set'], params['seq'])
    mots_instances_dir = '%s/%s/instances/%s' % (params['input_dir'], params['set'], params['seq'])
    # MOT output data directories
    mot_gt_dir = '%s/%s/gt' % (params['mot_save_dir'], params['seq'])
    mot_img_dir = '%s/%s/img' % (params['mot_save_dir'], params['seq'])
    # Create MOT outputdirectories and link image data
    os.makedirs(mot_gt_dir, exist_ok=True)
    os.symlink(os.getcwd() + '/' + mots_images_dir, mot_img_dir)
    # COCO output data directories
    coco_labels_dir = '%s/%s/labels' % (params['coco_save_dir'], params['set'])
    coco_images_dir = '%s/%s/images' % (params['coco_save_dir'], params['set'])
    # Create COCO output directories
    os.makedirs(coco_labels_dir, exist_ok=True)
    os.makedirs(coco_images_dir, exist_ok=True)
    # Initialize MOT(S) counters
    mask, track, frames = 0, 0, 0
    with open('%s/gt.txt' % mot_gt_dir, 'w') as gt_mot:
        for filename in sorted(os.listdir(mots_instances_dir)):
            image_mots_path = '%s/%s' % (mots_images_dir, filename)
            instance_mots_path = '%s/%s' % (mots_instances_dir, filename)
            # Skip if file not found
            if not os.path.isfile(image_mots_path): continue
            if not os.path.isfile(instance_mots_path): continue
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Determine COCO output paths
                name, ext = filename.split('.')
                image_coco_path = '%s/%s%s.%s' % (coco_images_dir, params['seq'], name, ext)
                label_coco_path = '%s/%s%s.%s' % (coco_labels_dir, params['seq'], name, "txt")
                # Link image data
                os.symlink(os.getcwd() + '/' + image_mots_path, image_coco_path)
                # Count frames
                with open(label_coco_path, 'w') as label_coco:
                    frames += 1
                    img = np.array(Image.open(os.path.join(mots_instances_dir, filename)))
                    obj_ids = np.unique(img)[1:]  # Exclude background
                    mask += len(obj_ids)
                    # Extract and log bounding boxes with track IDs
                    for obj_id in obj_ids:
                        obj_mask = (img == obj_id).astype(np.uint8)
                        # rle = encode_mask_to_RLE(obj_mask)
                        pos = np.where(obj_mask)
                        xmin = np.min(pos[1]) 
                        xmax = np.max(pos[1])
                        ymin = np.min(pos[0]) 
                        ymax = np.max(pos[0]) 
                        w = xmax - xmin
                        h = ymax - ymin
                        object_id = (obj_id%1000)+1
                        frame_id = int(filename.split(".")[0]) + 1
                        mask_h = 972
                        mask_w = 1296
                        # Normalize and center bounding boxes
                        nx = (xmin + xmax) / (2 * mask_w)
                        ny = (ymin + ymax) / (2 * mask_h)
                        nw = w / mask_w
                        nh = h / mask_h
                        # print(frame_id)
                        # f.write(f'{filename},{obj_id},{xmin},{ymin},{xmax},{ymax}\n')
                        # f.write(f'{frame_id},{object_id},{xmin},{ymin},{w},{h}, 1, {rle}\n')
                        # For MOTS eval tool
                        # f.write(f'{frame_id},{object_id}, 1, {mask_h}, {mask_w}, {rle}\n') 
                        # for TrackEval tool
                        gt_mot.write(f'{frame_id},{object_id},{xmin},{ymin},{w},{h},1,1,1\n') 
                        # for Ultralytics-COCO dataset # 47 is the class id for apple
                        label_coco.write("47 %.4f %.4f %.4f %.4f\n" % (nx, ny, nw, nh))
                    if np.max(obj_ids) > track:
                        track = np.max(obj_ids)                
            else:
                continue
    track_id = track % 1000
    ls = [frames, track_id, mask]
    # Create seqinfo.ini file
    with open('%s/seqinfo.ini' % mot_gt_dir[:-2], 'w') as seqinfo:
        seqinfo.write("[Sequence]\n")
        seqinfo.write("name=%s\n" % params['seq'])
        seqinfo.write("imDir=img\n")
        seqinfo.write("frameRate=30\n")
        seqinfo.write("seqLength=%d\n" % frames)
        seqinfo.write("imWidth=1296\n")
        seqinfo.write("imHeight=972\n")
        seqinfo.write("imExt=.%s\n" % ext)
    print("Done sequence %s" % params['seq'])
    return ls
