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
import pandas as pd
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
from detectron2.structures import BoxMode

from detectron2.data.datasets import register_coco_instances
register_coco_instances("AuroraDataset_train", {}, "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/_annotationsTrain.coco.json", "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/train")
register_coco_instances("AuroraDataset_val", {}, "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/_annotationsValid.coco.json", "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/valid")

##MODEL AND TRAINING
from detectron2.engine import DefaultTrainer
from detectron2.solver.build import get_default_optimizer_params

cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")) #RESNET50
cfg.merge_from_file("/home/lsquarzoni/work/RetinaNet/Script_Python/retinanet_mnv2.yaml") #MOBILENETV2
cfg.MODEL.WEIGHTS = "/home/lsquarzoni/work/RetinaNet/OpenImagesV6_TinCan/outputTraining1_100k_LR5e-5_MNV2pesante_OpenImagesTinCan/model_final.pth"

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TRAIN = ("AuroraDataset_train",)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00005  #5e-5
cfg.SOLVER.MAX_ITER = 2500   
cfg.SOLVER.WEIGHT_DECAY = 0.0001
#cfg.SOLVER.STEPS = []       
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
cfg.SOLVER.WARMUP_ITERS = 100 
cfg.MODEL.RETINANET.NUM_CLASSES = 2
#cfg.MODEL.BACKBONE.FREEZE_AT = 5
#cfg.MODEL.RESNETS.NORM = "BN"
#cfg.MODEL.RETINANET.NUM_CONVS = 1

#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.OUTPUT_DIR = "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/outputFineTuning1_2.5k_LR5e-5_MNV2pesante_TinCanAugmented"
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

##EVALUATION
from detectron2.evaluation import COCOEvaluator

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # path to the model we just trained
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
cfg.DATASETS.TEST = ("AuroraDataset_val",)

from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("AuroraDataset_val", output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "AuroraDataset_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))