#-*-coding:utf-8-*-
'''Encode target locations and labels.'''
from __future__ import print_function
import torch

import math
import itertools
import numpy as np

class DataEncoder:
    def __init__(self,mode = 'text'):
        '''Compute default box sizes with scale and aspect transform.'''
        #when mode is text,longer box will be initiated
        scale = 300.
        steps = [s / scale for s in (8, 16, 32, 64, 100, 300)]
        sizes = [s / scale for s in (30, 60, 111, 162, 213, 264, 315)]
        aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2,), (2,))
        feature_map_sizes = (38, 19, 10, 5, 3, 1)

        num_layers = len(feature_map_sizes)

        if(mode == 'text'):
            aspect_ratios = ((2,3,4), (2,3,4), (2,3,4), (2,3,4), (2,3), (2,3))
            boxes = []
            for i in range(num_layers):
                fmsize = feature_map_sizes[i]
                for h, w in itertools.product(range(fmsize), repeat=2):
                    cx = (w + 0.5) * steps[i]
                    cy = (h + 0.5) * steps[i]

                    s = sizes[i]
                    boxes.append((cx, cy, s, s))

                    s = math.sqrt(sizes[i] * sizes[i + 1])
                    boxes.append((cx, cy, s, s))

                    s = sizes[i]
                    for ar in aspect_ratios[i]:
                        boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))

        else:
            boxes = []
            for i in range(num_layers):
                fmsize = feature_map_sizes[i]
                for h, w in itertools.product(range(fmsize), repeat=2):
                    cx = (w + 0.5) * steps[i]
                    cy = (h + 0.5) * steps[i]

                    s = sizes[i]
                    boxes.append((cx, cy, s, s))

                    s = math.sqrt(sizes[i] * sizes[i + 1])
                    boxes.append((cx, cy, s, s))

                    s = sizes[i]
                    for ar in aspect_ratios[i]:
                        boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                        boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))




        self.default_boxes = torch.Tensor(boxes)

    def iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].

        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].

        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def encode(self, boxes, classes, threshold=0.5):
        '''Transform target bounding boxes and class labels to SSD boxes and classes.

        Match each object box to all the default boxes, pick the ones with the
        Jaccard-Index > 0.5:
            Jaccard(A,B) = AB / (A+B-AB)

        Args:
          boxes: (tensor) object bounding boxes (xmin,ymin,xmax,ymax) of a image, sized [#obj, 4].
          classes: (tensor) object class labels of a image, sized [#obj,].
          threshold: (float) Jaccard index threshold

        Returns:
          boxes: (tensor) bounding boxes, sized [#obj, 8732, 4].
          classes: (tensor) class labels, sized [8732,]
        '''
        default_boxes = self.default_boxes
        num_default_boxes = default_boxes.size(0)
        num_objs = boxes.size(0)

        iou = self.iou(  # [#obj,8732]
            boxes,
            torch.cat([default_boxes[:,:2] - default_boxes[:,2:]/2,
                       default_boxes[:,:2] + default_boxes[:,2:]/2], 1)
        )

        iou, max_idx = iou.max(0)  # [1,8732]
        max_idx.squeeze_(0)        # [8732,]
        iou.squeeze_(0)            # [8732,]

        boxes = boxes[max_idx]     # [8732,4]
        variances = [0.1, 0.2]     # 这个东西又什么用
        cxcy = (boxes[:,:2] + boxes[:,2:])/2 - default_boxes[:,:2]  # [8732,2]
        cxcy /= variances[0] * default_boxes[:,2:]
        wh = (boxes[:,2:] - boxes[:,:2]) / default_boxes[:,2:]      # [8732,2]
        wh = torch.log(wh) / variances[1]
        loc = torch.cat([cxcy, wh], 1)  # [8732,4]

        conf = 1 + classes[max_idx]   # [8732,], background class = 0
        conf[iou<threshold] = 0       # background
        return loc, conf

    def nms(self, bboxes, scores, threshold=0.5, mode='union'):
        '''Non maximum suppression.

        Args:
          bboxes: (tensor) bounding boxes, sized [N,4].
          scores: (tensor) bbox scores, sized [N,].
          threshold: (float) overlap threshold.
          mode: (str) 'union' or 'min'.

        Returns:
          keep: (tensor) selected indices.

        Ref:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        '''
        try:
            x1 = bboxes[:,0]
            y1 = bboxes[:,1]
            x2 = bboxes[:,2]
            y2 = bboxes[:,3]
        except Exception as e:
            print(bboxes)
            raise IndexError

        # 每一个候选框的面积
        areas = (x2-x1) * (y2-y1)
        # order是按照score降序排序的
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            if(order.size() == torch.Size([])):
                i = order.item()
            else:
                i = order[0].item()

            # 概率最大的保留
            keep.append(i)
            # 剩下最后一个则退出
            if order.numel() == 1:
                break
            # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            if mode == 'union':
                # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积)
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)
            # 找到重叠度不高于阈值的矩形框索引
            ids = (ovr<=threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids+1]   #加1的原因是上面是从order[1]算起的
        # print(keep)
        return torch.LongTensor(keep)

    def _decode(self, loc, conf,threshold=0.5):
        '''Transform predicted loc/conf back to real bbox locations and class labels.

        Args:
          loc: (tensor) predicted loc, sized [8732,4].
          conf: (tensor) predicted conf, sized [8732,6].

        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj,1].
        '''
        variances = [0.1, 0.2]
        wh = torch.exp(loc[:,2:]*variances[1]) * self.default_boxes[:,2:]
        cxcy = loc[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        boxes = torch.cat([cxcy-wh/2, cxcy+wh/2], 1)  # [8732,4]

        max_conf, labels = conf.max(dim=1)  # value:(8732,) index:(8732)
        #(8732,) (8732,) (7716, 1) 即非背景的有7716组
        # print('max_conf, labels, labels.nonzero size:',max_conf.size(),labels.size(),labels.nonzero().size())
        try:
            ids = labels.nonzero().squeeze()  # [#boxes,]
            # print(ids)
        except Exception as e:
            ids = torch.LongTensor([1])
        keep = self.nms(boxes[ids], max_conf[ids],threshold=threshold)# len(keep)=2005,NMS后还留下2005组
        # (2069, 4) (2069,) (2069,)
        # print(boxes[ids][keep].size(), labels[ids][keep].size(), max_conf[ids][keep].size())
        # print(boxes[ids][keep],labels[ids])

        # if boxes[ids][keep].size(0) == 1:
            # print(boxes[ids][keep])
            # return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]

        # print('fin :',boxes[ids][keep][0], labels[ids][keep][0], max_conf[ids][keep][0])
        # print(max_conf[ids][keep])
        # print(labels[ids][keep])
        # print(boxes[ids][keep])
        # _,index = max_conf[ids][keep].max()
        # print('max conf:\n',max_conf[ids][keep][index],'bbox:\n',boxes[ids][keep][index],'labels:\n',labels[ids][keep][index])
        print('pred label:',labels[ids][keep[0]])
        return boxes[ids][keep[0]], labels[ids][keep[0]], max_conf[ids][keep[0]]


    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        '''Transform predicted loc/conf back to real bbox locations and class labels.

        Args:
          loc: (tensor) predicted loc, sized [8732,4].
          conf: (tensor) predicted conf, sized [8732,6].

        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj,1].
        '''
        variances = (0.1, 0.2)
        xy = loc_preds[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        wh = torch.exp(loc_preds[:,2:]*variances[1]) * self.default_boxes[:,2:]
        box_preds = torch.cat([xy-wh/2, xy+wh/2], 1)  #to xmin,ymin,xmax,ymax

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1) # 6
        for i in range(num_classes-1): # (0,1,2,3,4)
            score = cls_preds[:,i+1]  # 0为背景，故取值(1,2,3,4,5)对应5个种类
            mask = score > score_thresh # tensor( dtype=torch.uint8)
            if not mask.any(): # 全为0 conf即均比阈值小
                continue
            # mask.nonzero().squeeze() 返回不为0的Index。取出对应的box和score
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]
            if len(box.size()) == 1:
                box =box.unsqueeze(0)
            keep = self.nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.LongTensor(len(box[keep])).fill_(i))
            scores.append(score[keep])

        # print(boxes, labels, scores)
        if(len(boxes) == 0):
            return None,None,None
        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)
        return boxes, labels, scores