from __future__ import print_function

import os
import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.ssd import SSD300
from Dataset.SynthDataset import ListDataset
from utils.encoder import DataEncoder

rootdir = 'G:/pycode/TextDetection/data/'
val_name = 'tiny_list_val1.txt'
train_name = 'tiny_list_train1.txt'
valpath = os.path.join(rootdir,val_name)
trainpath = os.path.join(rootdir,train_name)

mode = 'normal'
checkpointDir = 1

with open(trainpath) as f:
    train_lines = f.readlines()

with open(valpath) as f:
    val_lines = f.readlines()


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
data_encoder = DataEncoder(mode=mode)

batch_size = 1
num_workers = 1


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

trainset = ListDataset(root=rootdir, list_file=train_name, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

testset = ListDataset(root=rootdir, list_file=val_name, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

net = SSD300(mode=mode)
# net.load_state_dict(torch.load(checkpointDir,map_location = torch.device("cpu"))['net'])
net.load_state_dict(torch.load(checkpointDir,map_location = torch.device("cpu")))
net.to(device)


dataType = ['test','train']
dataloaders = {'train':trainloader,'test':testloader}
imagenameLines = {'train':train_lines,'test':val_lines}
resStore = {'train':'trainRes','test':'testRes'}


score_thresh = 0.2
nms_thresh = 0.2
font = cv2.FONT_HERSHEY_SIMPLEX
for dtp in dataType:
    print('Phase: {}'.format(dtp))
    storeDir = os.path.join(rootdir,resStore[dtp])
    if(not os.path.exists(storeDir)):
        os.makedirs(storeDir)

    for idx,(images, _, _) in enumerate(dataloaders[dtp]):
        with torch.no_grad():
            images = images.to(device)
            loc_preds, conf_preds = net(images)
            boxes, labels, scores = data_encoder.decode(loc_preds.cpu().data.squeeze(),
                                                        F.softmax(conf_preds.cpu().squeeze(), dim=1).data,
                                                        score_thresh=score_thresh, nms_thresh=nms_thresh)


        imageName = os.path.join(rootdir,imagenameLines[dtp][idx].split()[0])
        imgStoreName = os.path.join(storeDir, imageName.split('/')[-1])
        img = cv2.imread(imageName, 1)
        h, w, c = img.shape

        if(boxes is None):
            cv2.imwrite(imgStoreName, img)
            continue

        boxes = torch.clamp(boxes, 0.0, 1.0, out=None)
        boxes *= torch.Tensor([w, h, w, h]).expand_as(boxes)
        boxes = boxes.int()

        for i in range(boxes.size()[0]):
            x_y_min = (boxes[i][0].item(), boxes[i][1].item())
            x_y_max = (boxes[i][2].item(), boxes[i][3].item())
            img = cv2.rectangle(img, x_y_min, x_y_max, (0, 255, 0), 3)

            cv2.putText(img, str(round(scores[i].item(),2)), x_y_min, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA, bottomLeftOrigin=False)


        cv2.imwrite(imgStoreName,img)
