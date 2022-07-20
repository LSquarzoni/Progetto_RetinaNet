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
register_coco_instances("images_train", {}, "/scratch/datasets/coco/annotations/instances_train2014.json", "/scratch/datasets/coco/train2014")
register_coco_instances("images_val", {}, "/scratch/datasets/coco/annotations/instances_val2014.json", "/scratch/datasets/coco/val2014")

##MODEL AND TRAINING
from detectron2.engine import DefaultTrainer
from detectron2.solver.build import get_default_optimizer_params

cfg = get_cfg()

#Due modelli da trainare per procedere ad una comparazione:

#MODIFICA RESNET: nel file dentro model zoo base_retinanet.yaml ho commentato alcuni valori
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")) #RESNET50
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
cfg.merge_from_file("/home/lsquarzoni/work/RetinaNet/Script_Python/retinanet_mnv2.yaml") #MOBILENETV2
cfg.MODEL.WEIGHTS = "/home/lsquarzoni/work/RetinaNet/COCOdataset2014/outputTrain3.1.1.1_100kiter_LR0.00005_MNV2finale/model_final.pth"

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TRAIN = ("images_train",)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00005  #5e-5
cfg.SOLVER.MAX_ITER = 100000    
cfg.SOLVER.WEIGHT_DECAY = 0.0001
#cfg.SOLVER.STEPS = []       
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
cfg.SOLVER.WARMUP_ITERS = 100 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80 
cfg.MODEL.RETINANET.NUM_CLASSES = 80 
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.RESNETS.NORM = "BN"
cfg.MODEL.RETINANET.NUM_CONVS = 1 ##RIDUCO IL NUMERO DI CONV LAYER DELLA HEAD DA 4 A 1

##TRAIN WITH 2 GPUs:
""" /home/lsquarzoni/work/RetinaNet/detectron2/tools/train_net.py \
    --config-file /home/lsquarzoni/work/RetinaNet/detectron2/detectron2/model_zoo/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml \
    --num-gpus 2 SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 \
    SOLVER.MAX_ITER 1000 MODEL.ROI_HEADS.NUM_CLASSES 80 MODEL.RETINANET.NUM_CLASSES 80 \
    MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 128 MODEL.BACKBONE.FREEZE_AT 0 MODEL.RESNETS.NORM SyncBN """
""" Command: CUDA_VISIBLE_DEVICES=0,1 python file.py """
#cfg.SOLVER.REFERENCE_WORLD_SIZE = 2
#cfg.SOLVER.IMS_PER_BATCH = 8  #8 o 16, da provare
#cfg.MODEL.BACKBONE.FREEZE_AT = 0
#cfg.MODEL.RESNETS.NORM = "SyncBN"
#cfg.MODEL.RETINANET.NORM = "SyncBN"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#optimizer = optim.Adam(build_model(cfg).parameters() , lr=0.0005, betas=(0.9,0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

##EVALUATION
from detectron2.evaluation import COCOEvaluator

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # path to the model we just trained
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
cfg.DATASETS.TEST = ("images_val",)
#trainer.resume_or_load(resume=True)

""" evaluator = COCOEvaluator("images_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
trainer.test(cfg=cfg,
             model=trainer.model,
             evaluators=evaluator) """

from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("images_val", output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "images_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))