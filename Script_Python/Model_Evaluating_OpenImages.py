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

##MODEL
from detectron2.engine import DefaultTrainer
from detectron2_backbone import backbone
from detectron2_backbone.config import add_backbone_config
from detectron2_backbone.backbone.fpn import build_retinanet_mnv2_fpn_backbone
from detectron2.modeling import build_model

# Set up model and training configurations
cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")) #RESNET50
cfg.merge_from_file("/home/lsquarzoni/work/RetinaNet/Script_Python/retinanet_mnv2.yaml")
cfg.MODEL.WEIGHTS = "/home/lsquarzoni/work/RetinaNet/Aurora_Dataset/outputFineTuning1_TinCanAugmented_2.5k_LR5e-5_FreezedBN/model_final.pth"
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.RETINANET.NUM_CONVS = 1 
cfg.DATASETS.TEST = ("bottle_tin_can_val",)

# Specify class number
cfg.MODEL.RETINANET.NUM_CLASSES = len(target_classes) # if using RetinaNet

#EVALUATION
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("bottle_tin_can_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "bottle_tin_can_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
""" trainer = DefaultTrainer(cfg)
trainer.test(cfg=cfg,
             model=trainer.model,
             evaluators=evaluator) """