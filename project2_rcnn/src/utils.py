import cv2
import torch
import random
import numpy as np



def collate_fn(batch):
    # need to model
    return tuple(zip(*batch))


def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    
    if dx < 0:
        return 0.0
    
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area


def plot_rectangle(out: list, image: list, threshold: int, color: tuple, outline_thickness:int = 2, text:bool = False)->list:
    """
    out: list this out model, have tensor need convert to numpy
    image: list
    threshold: int prob threshold filter bad predict
    color: tuple color (255, 0,0)
    outline_thickness: int thickness lines
    text: str add text to image

    plot predict lines(box) on image    
    """   
    b = out[0]['boxes'].data.cpu().numpy()
    if len(b) > 0:
        s = out[0]['scores'].data.cpu().numpy()      
        bx = b[s>=threshold]
        if len(bx) > 0:
            b = bx[0]
            cv2.rectangle(image,
                         (b[0], b[1]),
                         (b[2], b[3]),
                         color,
                         outline_thickness)
            if text:
                cv2.putText(image,
                            str(round(s[0], 2)),
                            org=(int((b[0]+b[2])/2), 
                            int(b[1]- 40)),
                            fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            fontScale = 1.05,
                            color = (255, 255, 255),
                            thickness = outline_thickness)
    return image


def set_seed(seed=0):
    #set seed 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True