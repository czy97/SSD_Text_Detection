import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision

from .multibox_layer import MultiBoxLayer


class L2Norm2d(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm2d,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSD300(nn.Module):
    input_size = 300

    def __init__(self,init_param = True,mode = 'text'):
        super(SSD300, self).__init__()

        # model
        self.base = self.VGG16()
        self.norm4 = L2Norm2d(512,20)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

        # multibox layer
        self.multibox = MultiBoxLayer(mode)

        if(init_param):
            self.paramInit()

    def forward(self, x):
        hs = []
        h = self.base(x)
        hs.append(self.norm4(h))  # conv4_3

        h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)  # conv7

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)  # conv8_2

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)  # conv9_2

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)  # conv10_2

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)  # conv11_2

        loc_preds, conf_preds = self.multibox(hs)
        return loc_preds, conf_preds

    def VGG16(self):
        '''VGG16 layers.'''
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)

    def paramInit(self):
        vgg16_pretrained = torchvision.models.vgg16(pretrained=True)
        baseFeatureUpdate = False
        for i in range(23):
            if(hasattr(vgg16_pretrained.features[i], 'weight')):
                self.base[i].weight.data.copy_(vgg16_pretrained.features[i].weight.data)
                self.base[i].bias.data.copy_(vgg16_pretrained.features[i].bias.data)
                self.base[i].weight.requires_grad = baseFeatureUpdate
                self.base[i].bias.requires_grad = baseFeatureUpdate

        self.conv5_1.weight.data.copy_(vgg16_pretrained.features[24].weight.data)
        self.conv5_1.bias.data.copy_(vgg16_pretrained.features[24].bias.data)
        self.conv5_2.weight.data.copy_(vgg16_pretrained.features[26].weight.data)
        self.conv5_2.bias.data.copy_(vgg16_pretrained.features[26].bias.data)
        self.conv5_3.weight.data.copy_(vgg16_pretrained.features[28].weight.data)
        self.conv5_3.bias.data.copy_(vgg16_pretrained.features[28].bias.data)

        self.conv5_1.weight.requires_grad = baseFeatureUpdate
        self.conv5_1.bias.requires_grad = baseFeatureUpdate
        self.conv5_2.weight.requires_grad = baseFeatureUpdate
        self.conv5_2.bias.requires_grad = baseFeatureUpdate
        self.conv5_3.weight.requires_grad = baseFeatureUpdate
        self.conv5_3.bias.requires_grad = baseFeatureUpdate


        torch.nn.init.xavier_normal_(self.conv6.weight, gain=1)
        torch.nn.init.xavier_normal_(self.conv7.weight, gain=1)
        torch.nn.init.xavier_normal_(self.conv8_1.weight, gain=1)
        torch.nn.init.xavier_normal_(self.conv8_2.weight, gain=1)
        torch.nn.init.xavier_normal_(self.conv9_1.weight, gain=1)
        torch.nn.init.xavier_normal_(self.conv9_2.weight, gain=1)
        torch.nn.init.xavier_normal_(self.conv10_1.weight, gain=1)
        torch.nn.init.xavier_normal_(self.conv10_2.weight, gain=1)
        torch.nn.init.xavier_normal_(self.conv11_1.weight, gain=1)
        torch.nn.init.xavier_normal_(self.conv11_2.weight, gain=1)

        for i in range(5):
            torch.nn.init.xavier_normal_(self.multibox.conf_layers[i].weight, gain=1)
            torch.nn.init.xavier_normal_(self.multibox.loc_layers[i].weight, gain=1)

    def resetRequireGrad(self):
        baseFeatureUpdate = True
        for i in range(23):
            if(hasattr(self.base[i], 'weight')):
                self.base[i].weight.requires_grad = baseFeatureUpdate
                self.base[i].bias.requires_grad = baseFeatureUpdate

        self.conv5_1.weight.requires_grad = baseFeatureUpdate
        self.conv5_1.bias.requires_grad = baseFeatureUpdate
        self.conv5_2.weight.requires_grad = baseFeatureUpdate
        self.conv5_2.bias.requires_grad = baseFeatureUpdate
        self.conv5_3.weight.requires_grad = baseFeatureUpdate
        self.conv5_3.bias.requires_grad = baseFeatureUpdate







if __name__ == '__main__':
    """
    For test
    """
    model = SSD300()
    inp = torch.randn(2,3,300,300)
    loc_preds, conf_preds = model(inp)
    print("loc shape:\n",loc_preds.size()) # (N,8732, 4)
    print("loc :\n",loc_preds)
    print("conf shape:\n",conf_preds.size()) # (N, 8732, 6)
    print("conf :\n",conf_preds)