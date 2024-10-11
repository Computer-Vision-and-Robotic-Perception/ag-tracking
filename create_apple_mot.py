import os
import shutil
import multiprocessing
from lib.utils.data_utils import create_mot_n_coco_for_sequence

if __name__ == '__main__':
    # Data directories
    data_dir = 'data/AppleMOTS'
    mot_save_dir = 'data/AppleMOTS/MOT'
    coco_save_dir = 'data/AppleMOTS/COCO'
    # Get sequences
    seqs_train = os.listdir(data_dir + '/train/images')
    seqs_test = os.listdir(data_dir + '/testing/images')
    seqs = seqs_train + seqs_test
    # Set sequences
    set_seqs = ['train'] * len(seqs_train) + ['testing'] * len(seqs_test)
    seq_params = []
    # Create parameters for each sequence
    for seq, set_seq in zip(seqs, set_seqs):
        seq_params.append({
            'seq': seq,
            'set': set_seq,
            'input_dir': data_dir,
            'mot_save_dir': mot_save_dir,
            'coco_save_dir': coco_save_dir
        })
    # Clean directories
    if os.path.exists(mot_save_dir): shutil.rmtree(mot_save_dir)
    if os.path.exists(coco_save_dir): shutil.rmtree(coco_save_dir)
    # Create directories
    os.makedirs(mot_save_dir, exist_ok=True)
    os.makedirs(coco_save_dir, exist_ok=True)
    # Process sequences in parallel
    with multiprocessing.Pool(processes=12) as pool:
        pool.map(create_mot_n_coco_for_sequence, seq_params)

