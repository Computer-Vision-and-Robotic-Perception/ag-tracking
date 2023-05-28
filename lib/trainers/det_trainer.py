import os
import torch
from ..utils.det_utils import utils
from torch.utils.data import DataLoader
from ..models.frcnn_fpn import FRCNN_FPN
from ..utils.det_utils.engine import train_one_epoch, evaluate

class DetTrainer():
    def __init__(self, cfg, dset, name):
        self.dataset = dset
        self.num_epochs = cfg['epochs']
        self.checkpoint = cfg['load_checkpoint']
        self.save_path = cfg['outdir'] + 'det/' + name + '/' + cfg['backbone'] + '_weights'
        self.loader = DataLoader(dset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], collate_fn=utils.collate_fn)
        self.model = FRCNN_FPN(backbone_type=cfg['backbone'])
        if self.checkpoint:
            chkpt = self.save_path + '/model_epoch_%d.pt' % self.checkpoint
            print('loading checkpoint: "%s"' % chkpt)
            model_state_dict = torch.load(chkpt)
            self.model.load_state_dict(model_state_dict)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
        self.device = torch.device('cuda:0')
    
    def train(self):
        self.model.to(self.device)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        for epoch in range(self.checkpoint, self.num_epochs + 1):
            print('\n++++++++++++++++++++')
            print('>>> Epoch %03d ++++++' % (epoch))
            print('++++++++++++++++++++')
            train_one_epoch(self.model, self.optimizer, self.loader, self.device, epoch, print_freq=10)
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), self.save_path + "/model_epoch_{}.pt".format(epoch))
                evaluate(self.model, self.loader, self.device) # This evaluates in the trainign set

    def eval(self, path):
        # files = list(os.listdir(path))
        # print(files)

        # print('loading checkpoint: "%s"' % chkpt)
        # model_state_dict = torch.load(chkpt)
        # self.model.load_state_dict(model_state_dict)

        stat_description = ['APa', 'AP.5', 'AP.75', 'APs', 'APm', 'APl',
                            'AR1', 'AR10', 'AR100', 'APs', 'APm', 'APl']
        print(stat_description[0], stat_description[8])
        self.model.to(self.device)
        evaluator = evaluate(self.model, self.loader, self.device)
        stats = list(evaluator.coco_eval.values())[0].stats
        for desc, stat in zip(stat_description, stats):
            print(desc, stat, '*' if desc in ['APa', 'AR100'] else '')
