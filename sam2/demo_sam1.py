import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2


def build_sam():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg ="configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model,points_per_side=32,
       pred_iou_thresh=0.3,stability_score_thresh=0.7,min_mask_region_area=20)#default32 our 8
    return mask_generator

def SAM2(image,mask_generator):
    
    auto_mask=mask_generator.generate(image)
   
    
    return auto_mask



