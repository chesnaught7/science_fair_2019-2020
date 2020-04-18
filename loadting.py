from __future__ import print_function, division
from PIL import Image
from torchvision import transforms


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import time
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])
model_ft = models.resnet18(num_classes=3)
model_ft.load_state_dict(torch.load('./resnetmodel3.pth'))
model_ft.eval()
from PIL import Image
from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])
img = Image.open("../Downloads/Chest.jpeg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
out = model_ft(batch_t)
print(out.shape)
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

labels = ['cardiomegaly', 'pneumonia', 'nf']
print(labels[index[0]], percentage[index[0]].item())
# XXX:
