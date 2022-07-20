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

# Choose classes with > 900 annotations
target_classes = [
    'Bottle',
    'Tin can'
 ]

# Sort and create ID for each class
target_classes = sorted(target_classes)
class2id = {class_: id_ for id_, class_ in enumerate(target_classes)}
print(class2id)

# Get class names
classes = pd.read_csv("class-descriptions-boxable.csv", header=None, names=['LabelName', 'Class'])
subset_classes =classes[classes['Class'].isin(target_classes)]
subset_classes

# Prepare annotation files
for folder in ['train', 'validation']:
    # Load annotation files
    annot_df = pd.read_csv(f"{folder}-annotations-bbox.csv")
    # Inner join with subset_classes
    annot_df = annot_df.merge(subset_classes, on='LabelName')
    # Create `ClassID`
    annot_df['ClassID'] = annot_df['Class'].apply(lambda x: class2id[x])
    # Save subset files
    annot_df.to_csv(f"{folder}-annotations-bbox-target.csv", index=False)
    del annot_df

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

# Set up model and training configurations
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("bottle_tin_can_train",)
cfg.DATASETS.TEST = ("bottle_tin_can_val",)

# Training hyperparameters
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 #(default: 512)
cfg.SOLVER.BASE_LR = 5e-4
cfg.SOLVER.MAX_ITER = 100000 # 300 iterations seems good enough for this dataset; you will need to train longer for a practical dataset
cfg.SOLVER.WARMUP_ITERS = 100
cfg.SOLVER.STEPS = (200,)

# Specify class number
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(target_classes)
cfg.MODEL.RETINANET.NUM_CLASSES = len(target_classes) # if using RetinaNet

cfg.DATALOADER.NUM_WORKERS = 6
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Set up trainer
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

##EVALUATION

from detectron2.evaluation import COCOEvaluator

evaluator = COCOEvaluator("bottle_tin_can_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
trainer.test(cfg=cfg,
             model=trainer.model,
             evaluators=evaluator)

"""
##INFERENCE ON THE VALIDATION SET

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

dataset_dicts = get_detectron_dicts(val_df)

for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=bottle_tin_can_metadata, scale=0.8)
    pred = v.draw_instance_predictions(outputs["instances"][:2].to("cpu"))
    cv2_imshow(pred.get_image()[:, :, ::-1])
    plt.title("Prediction");
"""
