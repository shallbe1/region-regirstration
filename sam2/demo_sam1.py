import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import time
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0

    for ann in sorted_anns:
        #name=name+1
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        
    ax.imshow(img)
    

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



