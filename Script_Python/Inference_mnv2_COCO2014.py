#basic
import os, cv2, random, json
from os.path import join
from tqdm import tqdm
import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

#torch
import torch; print('\nPyTorch version in use:', torch.__version__, '\ncuda avail: ', torch.cuda.is_available())
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

#torchvision
import torchvision
from torchvision import transforms, datasets

#others
from copy import deepcopy
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: %s' % device)

# detectron2
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.modeling import build_model
from detectron2_backbone import backbone
from detectron2_backbone.config import add_backbone_config
from detectron2_backbone.backbone.fpn import build_retinanet_mnv2_fpn_backbone
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import sys
def cv2_imshow(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(), plt.imshow(im), plt.axis('off')

from detectron2.data.datasets import register_coco_instances
register_coco_instances("images_train", {}, "/scratch/datasets/coco/annotations/instances_train2014.json", "/scratch/datasets/coco/train2014")

#im_path = sys.argv[1]  #prende in input come argv[1] il path del'immagine 
#im = cv2.imread(im_path)

im = cv2.imread("/home/lsquarzoni/work/RetinaNet/Im_samples/persone.jpg") #path immagine

#Load model
cfg = get_cfg()
cfg.merge_from_file("/home/lsquarzoni/work/RetinaNet/Script_Python/retinanet_mnv2.yaml")
cfg.MODEL.WEIGHTS = "/home/lsquarzoni/work/RetinaNet/COCOdataset2014/output400000iter_MNV2_LR0.005/model_final.pth" # path to the model we just trained
cfg.DATASETS.TRAIN = ("images_train",)
cfg.MODEL.RETINANET.NUM_CLASSES = 80

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
print(cfg)
predictor = DefaultPredictor(cfg)
print(predictor.model)

##INFERENCE
outputs = predictor(im)
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.5)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite('/home/lsquarzoni/work/RetinaNet/Im_samples/Im_outputs/pred_mnv2_COCO.jpg', out.get_image()[:, :, ::-1])