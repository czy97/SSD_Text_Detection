from __future__ import print_function

import os
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models.ssd import SSD300
from Dataset.SynthDataset import ListDataset
from utils.loss import SSDLoss
from utils.encoder import DataEncoder


device = torch.device("cuda0" if torch.cuda.is_available() else "cpu")
data_encoder = DataEncoder()

batch_size = 1
num_workers = 1


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

trainset = ListDataset(root='G:/pycode/TextDetection/data/', list_file="tiny_list_train1.txt", train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

testset = ListDataset(root='G:/pycode/TextDetection/data/', list_file="tiny_list_val1.txt", train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

net = SSD300()
net.load_state_dict(torch.load('./checkpoint/ssd300_ckpt.pth',map_location = device)['net'])
net.to(device)

with torch.no_grad():
    images, _, _  = testset[0]
    images = images.unsqueeze(0)
    images = images.to(device)

    loc_preds, conf_preds = net(images)
    boxes, labels, scores = data_encoder.decode(loc_preds.cpu().data.squeeze(),
                                                F.softmax(conf_preds.cpu().squeeze(), dim=1).data,
                                                score_thresh=0.2, nms_thresh=0.2)

print(boxes)
print(labels)
print(scores)





