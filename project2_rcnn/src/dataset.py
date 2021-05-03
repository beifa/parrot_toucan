import torch
import numpy as np
from torch.utils.data import Dataset


class PT(Dataset):

    def __init__(self, data, transforms = None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
     
        dt = self.data[idx]        
        img = dt['image']        
        box_axis = dt['objects'][0]['points']['exterior']
        area = (box_axis[1][1] - box_axis[0][1]) * (box_axis[1][0] - box_axis[0][0])        
        boxes = torch.as_tensor(box_axis, dtype=torch.float32)
        boxes = boxes.reshape(-1, 4)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.ones((1), dtype = torch.long)
        target['image_id'] = torch.tensor([idx], dtype = torch.float32)
        target['area'] = torch.tensor(area, dtype = torch.float32)
        img = img / 255.0
        img = img.transpose(2, 0, 1).astype(np.float32)   
        if self.transforms is not None:
            img = self.transforms(img)   
        return torch.tensor(img, dtype = torch.float32), target 
        

class PT_test(Dataset):

    def __init__(self, data, transforms = None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
         
        img = self.data[idx]
        img = img / 255.0
        img = np.transpose(img, (2,0,1))
        if self.transforms is not None:
            img = self.transforms(img)    
        return torch.tensor(img, dtype = torch.float32), idx