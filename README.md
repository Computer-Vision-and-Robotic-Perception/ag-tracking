# Multi-Object Tracking in Agricultural Applications using a Vision Transformer for Spatial Association
This repository contains the implementation of the paper [Multi-Object Tracking in Agricultural Applications using a Vision Transformer for Spatial Association](https://www.sciencedirect.com/science/article/pii/S0168169924007701) [[Try this link](https://authors.elsevier.com/a/1jlNacFCSbHSL)]

## Quick Start

Instructions for Ubuntu 22.04 & Python 3.10

### Installation
1. Clone the repository, create a virtual environment, and install the package requirements
```bash
git clone https://github.com/Computer-Vision-and-Robotic-Perception/ag-tracking.git
cd ag-tracking
python3 -m venv venv
pip install -r requirements.txt
```
2. Follow the installation instructions for [Ultralytics](https://github.com/ultralytics/ultralytics.git) and [TrackEval](https://github.com/JonathonLuiten/TrackEval.git)

### Data Preparation
1. Download the LettuceMOT dataset from [here](https://mega.nz/folder/LlgByZ6Z#wmLa-TQ8NYGkPrJjJ5BfQw) to the `data/LettuceMOT` folder.
2. Download the AppleMOTS dataset from [here](https://git.wur.nl/said-lab/rt-obj-tracking) to the `data/AppleMOTS` folder. 
3. Create the `AppleMOT` dataset by running the following command:
```bash
python create_apple_mot.py
```

### Usage
1. Configure your task in `config/clasp_config.py`. The possible options for `task` are: 
   - `play` show the videos on screen (with annotations if available)
   - `det` run the detection pipeline
   - `slam` run the localization and mapping pipeline. Mainly for visualization purposes
   - `track` run the tracking pipeline composed of the FloraTracktor and LoFTR for localization
2. Configure the `mode` to be `train`, `eval` or `predict` according to the task you want to perform.
3. Run the script
```
python run.py
```

## Descriptions
- `config` contains the configuration files for the different tasks
    - `obj_config.py` is the base class for conguration files
    - `<dataset>_config.py` contains the configuration for the different datasets
    - `loftr_config.py` contains the configuration for the LoFTR model
    - `tracker_<type>_config.py` contains the configuration for the different tracking models (or versions described in the `version_note.txt` file)
    - `apples_coco.yaml` and `apples_bytetrack.yaml` are the configuration files for AppleMOTS with the Ultralytics framework.
- `lib/datasets/ag_mot.py` is a dataset class for MOT datasets and merged versions.
- `lib/models/tracktor/tracker_<type>.py` contains the implementation of each tracker.
- `lib/models/tracktor/slam.py` contains essentially the implementation of the spatial association module.
- `lib/models/frcnn_fpn.py` is the implementation of FasterRCNN with access to the backbone features and the regression head.
- `lib/models/tracker.py` is a general wrapper for the tracking models.
- `lib/trainers/frcnn_trainer.py` is the trainer for the detection models to use with Tracktors. COCO format is used for training.
- `lib/trainers/yolo_trainer.py` is the trainer for the detection models to use with Ultralytics. COCO format is used for training as detailed in the Ultralytics repository.
- `lib/utils/` 
    - `det_utils/` contains the utility functions for the FasterRCNN model training and evaluation using the COCO format.
    - `data_utils.py` contains utilities to load MOT labels and to create them from a COCO format.
    - `geom_utils.py` contains utilities for geometric transformations and operations.
    - `handle.py` (deprecated) contains utilities to handle TrackEval results and visualization.
    - `plots.py` contains utilities to create the plots for the paper.
    - `render_utils.py` contains utilities to `play`.
    - `track_eval.py` contains custom utilities to use TrackEval.


## Citation
1. Plain
```
Hernandez, Byron, and Medeiros, Henry. "Multi-Object Tracking in Agricultural Applications using a Vision Transformer for Spatial Association." Computers and Electronics in Agriculture 226 (2024): 109379.`
```

2. Bibtex
```
@article{hernandez2024floratracktor,
  title={Multi-Object Tracking in Agricultural Applications using a Vision Transformer for Spatial Association},
  author={Hernandez, Byron and Medeiros, Henry},
  journal={Computers and Electronics in Agriculture},
  volume={226},
  pages={109379},
  year={2024},
  publisher={Elsevier}
}
```
