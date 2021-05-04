import cv2
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import PIL.Image as Image
import albumentations as A
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from model import PT_RRCNN
from dataset import PT
from utils import collate_fn, calculate_iou, set_seed

import wandb
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--epoch', type=int, default = 35)
    parser.add_argument('--batch', type=int, default = 4)
    parser.add_argument('--n_workers', type=int, default = 4)
      
    args, _ = parser.parse_known_args()
    return args

def train(model, loader, optimizer):
    model.train()
    los = []  
    for images, targets in tqdm(loader):        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]       
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())   
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        los.append(losses.item()) 
    return los


def valid(model, loader):
    model.eval()
    scores = []
    iou = []
    with torch.no_grad():
        for images, targets in tqdm(loader): 
          images = list(image.to(device) for image in images)
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets] 
          loss_dict = model(images, targets)
          if len(loss_dict[0]['boxes']) > 0:
              bbox = loss_dict[0]['boxes'][0].data.cpu().numpy()
              ori_bbox = targets[0]['boxes'][0].data.cpu().numpy()
              score = loss_dict[0]['scores'][0].cpu().numpy()
              iou_calc = calculate_iou(ori_bbox, bbox)
          else:
              iou_calc = 0 
              score = 0
          iou.append(iou_calc)
          scores.append(score)
    return scores, iou

def tr_transform():
  return A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.2,
                                               p=0.9),
                    ],
                   bbox_params=A.BboxParams(format='pascal_voc', 
                                            label_fields=['labels']
                                            ),
                   )

def vl_transform():  
  return A.Compose([],
                   bbox_params=A.BboxParams(format='pascal_voc',  
                                            label_fields=['labels']
                                            ),
                   )


def showtime(model, train_data:list, fold:int,  transform:bool = None)->None:
    print(f'Fold {fold}')

    tr = np.take(train_data, tr_idx[fold])
    vl = np.take(train_data, vl_idx[fold])

    print(f'Train fold shape: {len(tr)}, Val: {len(vl)}')

    tr_dataset = PT(tr, tr_transform())
    tr_loader = DataLoader(tr_dataset,
                           batch_size = args.batch,
                           shuffle = True,
                           num_workers = args.n_workers,
                           collate_fn = collate_fn)
    
    vl_dataset = PT(vl, vl_transform())
    vl_loader = DataLoader(vl_dataset,
                           batch_size = 1,                     
                           num_workers = args.n_workers,          
                           collate_fn = collate_fn)

    model.to(device)

    if config.optimizer=='sgd':
      optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=0.9,weight_decay=0.005)
    elif config.optimizer=='adam':
      optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    if config.lr_scheduler is not None:
      lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='max',
                                                              factor=0.75,
                                                              patience=7)
    else:
      lr_scheduler = None

    best_iou = 0 
    wandb.watch(model) 
    for epoch in range(args.epoch):
        tr_los = train(model, tr_loader, optimizer)
        score, iou = valid(model, vl_loader)
        iou, score = np.mean(iou), np.mean(score)
        print(f"Epoch #{epoch}, Train loss: {np.mean(tr_los)} <--> Val scores: {score} <--> iou: {iou}")
        wandb.log({'epoch': epoch, "tr_loss": np.mean(tr_los), 'vl scores': score, 'iou': iou})
        if iou > best_iou:
            print(f'Save iou: {iou}')
            # torch.save(model.state_dict(), f'../project2_rcnn/model_rcnn/tmp/test_works_script_{f}.pth')   
            best_iou = iou
        if lr_scheduler is not None:
            lr_scheduler.step()
    wandb.finish()


if __name__ == "__main__":      
 
    PATH_ANOT = '../project2_rcnn/input/train_data/ann'
    PATH_IMG = '../project2_rcnn/input/train_data/img'
    set_seed(13)
    args = parse_args()
    train_data, tags = [], []  
    for f, img in zip(sorted(Path(PATH_ANOT).glob('*.*')), sorted(Path(PATH_IMG).glob('*.*'))):
        with Path(f).open() as json_file:
            data = json.load(json_file)
            data['image'] = cv2.cvtColor( cv2.imread(str(img)), cv2.COLOR_BGR2RGB)  
            train_data.append(data)
            if data['tags'] != []:
                tags.append(int(data['tags'][0]['name']))
            else: print(f)
    print('Train data size: ', len(train_data), len(tags))
    """
    what is tags:
        1 - big parrot proportion image
        2 - medium
        3 - small
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    tr_idx = []
    vl_idx = []
    for f, (tr, vl) in enumerate(skf.split(train_data, tags)):
        print(len(tr), len(vl))
        tr_idx.append(tr)
        vl_idx.append(vl)

    EXP_NAME = f"frcnn_v2_{args.lr}_{args.batch}, ver_{np.random.randint(111111111111111111)}"
    for f in range(5):

        wandb.init(project="parrot", name=f"{EXP_NAME}_f{f}")

        config = wandb.config
        config.exp_name = EXP_NAME
        config.learning_rate = args.lr#, 0.0025
        config.epoch = args.epoch  
        config.batch_size = args.batch
        config.lr_scheduler = None
        config.optimizer = 'sgd'

        model = PT_RRCNN()
        showtime(model, train_data, f)