from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable


class MultiBoxLayer(nn.Module):
    num_classes = 2
    num_coor = 4
    num_anchors = [4,6,6,6,4,4]
    in_planes = [512,1024,512,256,256,256]

    def __init__(self,mode = 'text'):
        super(MultiBoxLayer, self).__init__()

        if(mode == 'text'):
            self.num_anchors = [5, 5, 5, 5, 4, 4]
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.in_planes)):
            # Regression
        	self.loc_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*self.num_coor, kernel_size=3, padding=1))
            # classification
        	self.conf_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*self.num_classes, kernel_size=3, padding=1))

    def forward(self, xs):
        '''
        Args:
          xs: (list) of tensor containing intermediate layer outputs.

        Returns:
          loc_preds: (tensor) predicted locations, sized [N,8732,4].
          conf_preds: (tensor) predicted class confidences, sized [N,8732,21].
        '''
        y_locs = []
        y_confs = []
        for i,x in enumerate(xs):
            y_loc = self.loc_layers[i](x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0,2,3,1).contiguous()
            y_loc = y_loc.view(N,-1,self.num_coor)
            y_locs.append(y_loc)

            y_conf = self.conf_layers[i](x)
            y_conf = y_conf.permute(0,2,3,1).contiguous()
            y_conf = y_conf.view(N,-1,self.num_classes)
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)
        return loc_preds, conf_preds