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
import copy
from models.ssd import SSD300
from Dataset.SynthDataset import ListDataset
from utils.loss import SSDLoss
from utils.encoder import DataEncoder

parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--num_epochs', default=100, type=int, help='num_epochs')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
parser.add_argument('--gpu_id', default=0, type=int, help='gpu_id')
parser.add_argument('--box_type', required = True, help='box_type')
parser.add_argument('--data_dir', required = True, help='data_dir')
parser.add_argument('--storeParamName', required = True, help='storeParamName')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataParaller', '-d', action='store_true', help='dataParaller')


args = parser.parse_args()

if(args.dataParaller):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    cudaName = "cuda:"+str(args.gpu_id)
    device = torch.device(cudaName if torch.cuda.is_available() else "cpu")



best_loss = float('inf')  # best test loss
best_model_wts =None
start_epoch = 0  # start from epoch 0 or last epoch
batch_size = args.batchsize
num_workers = args.num_workers
end_epoch = args.num_epochs

save_checkpoint = True
# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

trainset = ListDataset(root=args.data_dir, list_file="tiny_list_train1.txt", train=True, transform=transform,mode=args.box_type)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset = ListDataset(root=args.data_dir, list_file="tiny_list_val1.txt", train=False, transform=transform,mode=args.box_type)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# Model
net = SSD300(mode=args.box_type).to(device)
# if args.resume:
#     print('====> Resuming from checkpoint..\n')
#     checkpoint = torch.load('./checkpoint/ssd300_ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_loss = checkpoint['loss']
#     start_epoch = checkpoint['epoch']
# else:
#     # Convert from pretrained VGG model.
#     net.load_state_dict(torch.load("./pretrained/ssd.pth"))

criterion = SSDLoss(num_classes=2)

if(args.dataParaller):
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.4 * end_epoch), int(0.7 * end_epoch),int(0.8 * end_epoch),int(0.9 * end_epoch)], gamma=0.1)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(trainloader):
        images = images.to(device)

        loc_targets = loc_targets.to(device)  # (N, 8732, 4)
        conf_targets = conf_targets.to(device)  # (N, 8732)

        optimizer.zero_grad()
        # (N,8732, 4) (N, 8732, 6)
        loc_preds, conf_preds = net(images)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
    print('Train data average loss/batch: %.4f' % (train_loss / (batch_idx + 1)))



def test(epoch):
    # print('\nTest')
    net.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, loc_targets, conf_targets) in enumerate(testloader):
            images = images.to(device)
            #  (N, 8732, 4) (N, 8732)
            loc_targets, conf_targets = loc_targets.to(device), conf_targets.to(device)
            # (N,8732, 4) (N, 8732, 6)
            loc_preds, conf_preds = net(images)

            loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
            test_loss += loss.data.item()
            total += images.size(0)
        print('Val data average loss/batch: %.4f'% (test_loss / (batch_idx + 1)))


    global best_loss,best_model_wts

    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Better loss:{}'.format(test_loss))
        if (args.dataParaller):
            best_model_wts = copy.deepcopy(net.module.state_dict())
        else:
            best_model_wts = copy.deepcopy(net.cpu().state_dict())
            net.to(device)

        best_loss = test_loss

label = 0
for epoch in range(start_epoch, start_epoch+end_epoch):
    if(label == 0 and epoch > int(end_epoch*0.2)):
        # net.resetRequireGrad()
        if (args.dataParaller):
            net.module.resetRequireGrad()
        else:
            net.resetRequireGrad()
        label = 1
    train(epoch)
    test(epoch)
    scheduler.step()

torch.save(best_model_wts, args.storeParamName)