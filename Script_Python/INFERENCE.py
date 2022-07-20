#basic
import os, cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

#torch
import torch; print('\nPyTorch version in use:', torch.__version__, '\ncuda avail: ', torch.cuda.is_available())

#others
from copy import deepcopy
from tqdm.autonotebook import tqdm
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: %s' % device)

# detectron2
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2_backbone import backbone
from detectron2_backbone.config import add_backbone_config
from detectron2_backbone.backbone.fpn import build_retinanet_mnv2_fpn_backbone
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
setup_logger()

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

#PRENDE IL PRIMO ARGOMENTO PASSATO DA LINEA DI COMANDO COME PERCORSO DELL'IMMAGINE SU CUI FARE INFERENZA
im_path = sys.argv[1]  #prende in input come argv[1] il path del'immagine 
im = cv2.imread(im_path)

#NEL CASO IN CUI SI VOGLIA DARE L'INDIRIZZO DELL'IMMAGINE DA SCRIPT
#im = cv2.imread("  ") #path immagine

#Load model
cfg = get_cfg()

#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")) #RESNET50
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")

cfg.merge_from_file("/home/lsquarzoni/work/RetinaNet/Script_Python/retinanet_mnv2.yaml") #MNV2
cfg.MODEL.WEIGHTS = "/home/lsquarzoni/work/RetinaNet/OpenImagesV6_TinCan/outputTraining1_OpIm+Drone_100k_LR5e-5_MNV2finale/model_final.pth" # path to the model we just trained

cfg.DATASETS.TRAIN = ("bottle_tin_can_train",)
cfg.MODEL.RETINANET.NUM_CLASSES = 2
cfg.MODEL.RETINANET.NUM_CONVS = 1

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4 #SOGLIA DI CONFIDENZA SOTTO LA QUALE LE PREDIZIONI VENGONO CONSIDERATE SBAGLIATE
predictor = DefaultPredictor(cfg)
print(predictor.model)

##INFERENCE
outputs = predictor(im)
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.5)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

#CREA UN'IMMAGINE DI OUTPUT CON LE BBOX AL PERCORSO ASSOLUTO SPECIFICATO COME PRIMO ARGOMENTO
cv2.imwrite('  ', out.get_image()[:, :, ::-1])