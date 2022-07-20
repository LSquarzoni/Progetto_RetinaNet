#basic
import os
from os.path import join
import numpy as np
from tqdm import tqdm
import time

#plotting
import matplotlib.pyplot as plt

#torch
import torch; print('\nPyTorch version in use:', torch.__version__, '\ncuda avail: ', torch.cuda.is_available())
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

#torchvision
import torchvision
from torchvision import transforms, datasets

# others
from copy import deepcopy
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: %s' % device)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.modeling import build_model

from detectron2_backbone import backbone
from detectron2_backbone.config import add_backbone_config

from detectron2_backbone.backbone.fpn import build_retinanet_mnv2_fpn_backbone
from detectron2 import model_zoo

cfg = get_cfg()
 # add config to detectron2

##cfg.merge_from_file("/home/lsquarzoni/work/RetinaNet/detectron2/detectron2/model_zoo/configs/COCO-Detection/retinanet_R_18_FPN.yaml")
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")) #RESNET50
cfg.merge_from_file("/home/lsquarzoni/work/RetinaNet/Script_Python/retinanet_mnv2.yaml")
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.RETINANET.NUM_CONVS = 1 ##RIDUCO IL NUMERO DI CONV LAYER DELLA HEAD DA 4 A 1
cfg.MODEL.RETINANET.NUM_CLASSES = 2

model = build_model(cfg)

print(model)
#print(cfg)


""" from torchinfo import summary

summary(model, (1, 1, 240, 320)) """



""" CONTEGGIO E TABULAZIONE PARAMETRI """

from detectron2.utils.analysis import parameter_count, parameter_count_table

tab = parameter_count_table(model, 2)
print(tab)



""" from prettytable import PrettyTable
    
    def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
count_parameters(model) """
