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
    del annot_df  """

train_df = pd.read_csv("train-annotations-bbox-target.csv")
val_df = pd.read_csv("validation-annotations-bbox-target.csv")

data_size = pd.concat([train_df['Class'].value_counts(), val_df['Class'].value_counts()], axis=1)
data_size.columns = ["Train", "Val"]
data_size

train_df['ImageID'].nunique(), val_df['ImageID'].nunique()

# Register dataset with Detectron2
print("Registering Datasets...")
DatasetCatalog.register("Bottle_TinCan_train", lambda d=train_df: get_detectron_dicts(d))
MetadataCatalog.get("Bottle_TinCan_train").set(thing_classes=target_classes)
DatasetCatalog.register("Bottle_TinCan_val", lambda d=val_df: get_detectron_dicts(d))
MetadataCatalog.get("Bottle_TinCan_val").set(thing_classes=target_classes)
print("Done!")

# Get metadata. It helps show class labels when we visualize bounding boxes
bottle_tincan_metadata = MetadataCatalog.get("Bottle_TinCan_train")

from detectron2.data.datasets import register_coco_instances
register_coco_instances("AuroraDataset_train", {}, "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/_annotationsTrain.coco.json", "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/train")
register_coco_instances("AuroraDataset_val", {}, "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/_annotationsValid.coco.json", "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/valid")

##MODEL AND TRAINING
from detectron2.engine import DefaultTrainer
from detectron2.solver.build import get_default_optimizer_params

cfg = get_cfg()

#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")) #RESNET50
cfg.merge_from_file("/home/lsquarzoni/work/RetinaNet/Script_Python/retinanet_mnv2.yaml") #MOBILENETV2
cfg.MODEL.WEIGHTS = "/home/lsquarzoni/work/RetinaNet/COCOdataset2014/outputTrain3.1.1.1_100kiter_LR0.00005_MNV2finale/model_final.pth"

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TRAIN = ("AuroraDataset_train", "Bottle_TinCan_train",)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00005  #5e-5
cfg.SOLVER.MAX_ITER = 100000    
cfg.SOLVER.WEIGHT_DECAY = 0.0001
#cfg.SOLVER.STEPS = []       
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
cfg.SOLVER.WARMUP_ITERS = 100 
cfg.MODEL.RETINANET.NUM_CLASSES = 2
#cfg.MODEL.BACKBONE.FREEZE_AT = 0
#cfg.MODEL.RESNETS.NORM = "BN"
cfg.MODEL.RETINANET.NUM_CONVS = 1 ##RIDUCO IL NUMERO DI CONV LAYER DELLA HEAD DA 4 A 1

#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.OUTPUT_DIR = "/home/lsquarzoni/work/RetinaNet/OpenImagesV6/outputTraining1_OpIm+Drone_100k_LR5e-5_MNV2finale"
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
evaluator = COCOEvaluator(cfg.DATASETS.TEST, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST)
print(inference_on_dataset(predictor.model, val_loader, evaluator))