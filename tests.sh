#!/bin/bash

- Test : Check play mode

add/change modes: 
  - Existing: 'play'  'train'*  'eval'*  'track'* 'slam'*
  - Change: ('train' 'det-train')* ('eval' 'det-test')*
            ('track' 'track-test')* ('slam' 'slam-test')*
  - New: 'det-predict'
         'track-train' 'track-test' 'train-predict'
         'slam-train' 'slam-test' 'slam-predict'

Potential modes: 'play'
                  'det-train'   'det-test'   'det-predict'
                  'slam-train'  'slam-test'  'slam-predict'
                  'track-train' 'track-test' 'track-predict'

- Test : Check det-train mode 
- Test : Check det-test mode
- Test : Check det-predict mode
- Test : Check slam-train mode
- Test : Check slam-test mode
- Test : Check slam-predict mode &probably not be needed
- Test : Check track-train mode
- Test : Check track-test mode
- Test : Check track-predict mode &probably not be needed 

define config mode: play 
define config task: train test predict


Dataloader: play, slam, det, track  # task
Model: train, test, predict         # mode
