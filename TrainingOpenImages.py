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

# others
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

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

#ESEGUIRE: cd /home/lsquarzoni/work/RetinaNet/OpenImagesV6
#          /home/lsquarzoni/work/RetinaNet/Script_Python/Training_OpenImagesV6_RetinaNet_Mnv2_10Mparam.py > /home/lsquarzoni/work/RetinaNet/Training/*FILE*

#USEFUL DETECTRON2 METHODS

def denormalize_bboxes(bboxes, height, width):
    """Denormalize bounding boxes in format of (xmin, ymin, xmax, ymax)."""
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height
    return np.round(bboxes)

def get_detectron_dicts(annot_df):
    """
    Create Detectron2's standard dataset from an annotation file.
    
    Args:
        annot_df (pd.DataFrame): annotation dataframe.
    Return:
        dataset_dicts (list[dict]): List of annotation dictionaries for Detectron2.
    """
    # Get image ids
    img_ids = annot_df["ImageID"].unique().tolist()
    
    dataset_dicts = []
    for img_id in tqdm(img_ids):
        file_name = f'images/{img_id}.jpg'
        if not os.path.exists(file_name): continue
        height, width = cv2.imread(file_name).shape[:2]
            
        record = {}
        record['file_name'] = file_name
        record['image_id'] = img_id
        record['height'] = height
        record['width'] = width
        
        # Extract bboxes from annotation file
        bboxes = annot_df[['XMin', 'YMin', 'XMax', 'YMax']][annot_df['ImageID'] == img_id].values
        bboxes = denormalize_bboxes(bboxes, height, width)
        class_ids = annot_df[['ClassID']][annot_df['ImageID'] == img_id].values
        
        annots = []
        for i, bbox in enumerate(bboxes.tolist()):
            annot = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(class_ids[i]),
            }
            annots.append(annot)

        record["annotations"] = annots
        dataset_dicts.append(record)
    return dataset_dicts

##DATASET AND DATALOADING

target_classes = [
    'Bottle',
    'Tin can'
 ]

len(target_classes)

# Sort and create ID for each class
target_classes = sorted(target_classes)
class2id = {class_: id_ for id_, class_ in enumerate(target_classes)}
print(class2id)

# Get class names
classes = pd.read_csv("class-descriptions-boxable.csv", header=None, names=['LabelName', 'Class'])
subset_classes =classes[classes['Class'].isin(target_classes)]
subset_classes

""" # Prepare annotation files
for folder in ['train', 'validation']:
    # Load annotation files
    annot_df = pd.read_csv(f"{folder}-annotations-bbox.csv")
    # Inner join with subset_classes
    annot_df = annot_df.merge(subset_classes, on='LabelName')
    # Create `ClassID`
    annot_df['ClassID'] = annot_df['Class'].apply(lambda x: class2id[x])
    # Save subset files
    annot_df.to_csv(f"{folder}-annotations-bbox-target.csv", index=False)
    del annot_df """

train_df = pd.read_csv("train-annotations-bbox-target.csv")
val_df = pd.read_csv("validation-annotations-bbox-target.csv")

data_size = pd.concat([train_df['Class'].value_counts(), val_df['Class'].value_counts()], axis=1)
data_size.columns = ["Train", "Val"]
data_size

train_df['ImageID'].nunique(), val_df['ImageID'].nunique()

# Register dataset with Detectron2
print("Registering Datasets...")
DatasetCatalog.register("bottle_tin_can_train", lambda d=train_df: get_detectron_dicts(d))
MetadataCatalog.get("bottle_tin_can_train").set(thing_classes=target_classes)
DatasetCatalog.register("bottle_tin_can_val", lambda d=val_df: get_detectron_dicts(d))
MetadataCatalog.get("bottle_tin_can_val").set(thing_classes=target_classes)
print("Done!")

# Get metadata. It helps show class labels when we visualize bounding boxes
bottle_tin_can_metadata = MetadataCatalog.get("bottle_tin_can_train")

##MODEL AND TRAINING

from detectron2.engine import DefaultTrainer
from detectron2_backbone import backbone
from detectron2_backbone.config import add_backbone_config
from detectron2_backbone.backbone.fpn import build_retinanet_mnv2_fpn_backbone
from detectron2.modeling import build_model
from detectron2.solver.build import get_default_optimizer_params

# Set up model and training configurations
cfg = get_cfg()
cfg.merge_from_file("/home/lsquarzoni/work/RetinaNet/Script_Python/retinanet_mnv2.yaml") #MNV2
cfg.MODEL.WEIGHTS = "/home/lsquarzoni/work/RetinaNet/COCOdataset2014/outputTrain3.1.1.1_100kiter_LR0.00005_MNV2finale/model_final.pth"
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")) #RESNET50
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model

cfg.DATASETS.TRAIN = ("bottle_tin_can_train",)
cfg.DATASETS.TEST = ()

# Training hyperparameters
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00005  #5e-5
cfg.SOLVER.MAX_ITER = 100000     
cfg.SOLVER.WEIGHT_DECAY = 0.0001
#cfg.SOLVER.STEPS = []       
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
cfg.SOLVER.WARMUP_ITERS = 100  
#cfg.MODEL.BACKBONE.FREEZE_AT = 0
#cfg.MODEL.RESNETS.NORM = "BN"
cfg.MODEL.RETINANET.NUM_CONVS = 1 ##RIDUCO IL NUMERO DI CONV LAYER DELLA HEAD DA 4 A 1

# Specify class number
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(target_classes)
cfg.MODEL.RETINANET.NUM_CLASSES = len(target_classes) # if using RetinaNet

#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.OUTPUT_DIR = "/home/lsquarzoni/work/RetinaNet/OpenImagesV6/outputTraining1_100k_LR5e-5_RandRotation_MNV2finale_OpenImages"

# Set up trainer
#optimizer = optim.Adam(cfg(), lr=0.0005, betas=(0.9,0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

##EVALUATION
from detectron2.evaluation import COCOEvaluator

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # path to the model we just trained
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
cfg.DATASETS.TEST = ("bottle_tin_can_val",)

evaluator = COCOEvaluator("bottle_tin_can_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
trainer.test(cfg=cfg,
             model=trainer.model,
             evaluators=evaluator)
