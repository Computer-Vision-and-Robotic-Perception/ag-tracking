# Agriculture MOT: Towards Multi Object Tracking in Agriculture
This repository contains the main implementation for the paper [Multi-Object Tracking in Agricultural Applications using a Vision Transformer for Spatial Association](https://www.sciencedirect.com/science/article/pii/S0168169924007701) [[Try this link](https://authors.elsevier.com/a/1jlNacFCSbHSL)]

## Quick Start

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
Create the `AppleMOT` dataset by running the following command:
```bash
python create_apple_mot.py
```

### Running
1. Configure your task in `config/clasp_config.py`. The possible options for `task` are: 
   1. `play` show the videos on screen (with annotations if available)
   2. `det` run the detection pipeline
   3. `slam` run the localization and mapping pipeline. Mainly for visualization purposes
   4. `track` run the tracking pipeline composed of the FloraTracktor and LoFTR for localization
2. Configure the `mode` to be `train`, `eval` or `predict` according to the task you want to perform.
3. Run the script
```
python run.py
```

## Descriptions

## Citation
1. Plain
```
Hernandez, Byron, and Henry Medeiros. "Multi-Object Tracking in Agricultural Applications using a Vision Transformer for Spatial Association." Computers and Electronics in Agriculture 226 (2024): 109379.`
```

2. Bibtex
```
@article{hernandez2024multi,
  title={Multi-Object Tracking in Agricultural Applications using a Vision Transformer for Spatial Association},
  author={Hernandez, Byron and Medeiros, Henry},
  journal={Computers and Electronics in Agriculture},
  volume={226},
  pages={109379},
  year={2024},
  publisher={Elsevier}
}
```
