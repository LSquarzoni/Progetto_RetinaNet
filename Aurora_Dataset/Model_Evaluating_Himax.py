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
register_coco_instances("AuroraDataset_val", {}, "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/_annotationsValid.coco.json", "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/valid")

##MODEL
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")) #RESNET50
cfg.merge_from_file("/home/lsquarzoni/work/RetinaNet/Script_Python/retinanet_mnv2.yaml") #MOBILENETV2
cfg.MODEL.WEIGHTS = "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/outputFineTuning1_2.5k_LR5e-5_MNV2finale/model_final.pth"

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("AuroraDataset_val",)
cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.RETINANET.NUM_CLASSES = 2
cfg.MODEL.RETINANET.NUM_CONVS = 1

#EVALUATION
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("AuroraDataset_val", output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "AuroraDataset_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))