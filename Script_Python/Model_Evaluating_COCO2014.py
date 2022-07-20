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
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
# import some common detectron2 utilities
from detectron2.modeling import build_model
from detectron2_backbone import backbone
from detectron2_backbone.config import add_backbone_config
from detectron2_backbone.backbone.fpn import build_retinanet_mnv2_fpn_backbone

from detectron2.data.datasets import register_coco_instances
register_coco_instances("images_val", {}, "/scratch/datasets/coco/annotations/instances_val2014.json", "/scratch/datasets/coco/val2014")

##MODEL
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
#Due modelli da trainare per procedere ad una comparazione:

#MODIFICA RESNET: nel file dentro model zoo base_retinanet.yaml ho commentato alcuni valori
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")) #RESNET50
cfg.merge_from_file("/home/lsquarzoni/work/RetinaNet/Script_Python/retinanet_mnv2.yaml") #MOBILENETV2
#cfg.MODEL.BACKBONE.FREEZE_AT = 0
#cfg.MODEL.RESNETS.NORM = "BN"

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "/home/lsquarzoni/work/RetinaNet/COCOdataset2014/output200000iter_MNV2_LR0.02/model_final.pth" # path to the model we just trained
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")

cfg.DATASETS.TEST = ("images_val",)
cfg.DATALOADER.NUM_WORKERS = 6

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80 
cfg.MODEL.RETINANET.NUM_CLASSES = 80 

#EVALUATION

""" from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

#trainer = DefaultTrainer(cfg)
#trainer.resume_or_load(resume=True)
model = build_model(cfg)
model.eval()
#predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("images_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "images_val")
print(inference_on_dataset(model, val_loader, evaluator)) """

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("images_val", output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "images_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))