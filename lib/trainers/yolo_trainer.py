import os
import sys
import shutil
import datetime
from ultralytics import YOLO
from ultralytics import settings


class UltraTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        settings.reset()
        settings.update({'datasets_dir': cfg['datadir'], 'runs_dir': cfg['outdir']})
        self.model = YOLO(cfg['yolo_maker'], verbose=True)  # build a new model from YAML file
    
    def train(self):
        self.model.train(data=self.cfg['yolo_data'], 
                         epochs=self.cfg['epochs'], 
                         batch=self.cfg['batch_size'],
                         imgsz=640, 
                         save_period=1,
                         device=self.cfg['device'])

    def eval(self):
        warn = ""
        if self.cfg['yolo_maker'] == 'yolov8n.pt': warn = "WARNING:"
        print(warn + " Using checkpoint:", self.cfg['yolo_maker'])
        self.model.val()

    def predict(self, max=50, img=None):
        if img:
            self.model(img)
        else:
            for i, file in enumerate(os.listdir(self.cfg['datadir'] + 'testing/images/')):
                if i == max: break
                if file.endswith('.jpg') or file.endswith('.png'):
                    self.model(self.cfg['datadir'] + 'testing/images/' + file, save=True, verbose=False)
        print('Results saved to', settings('runs_dir'))
