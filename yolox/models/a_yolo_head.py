#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, meshgrid

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv, RepVGGBlock




class AYOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        #每个anchor预测多少个框
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems_2 = nn.ModuleList()
        self.stems_3 = nn.ModuleList()
        self.reg_weights = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                    BaseConv(
                        in_channels=int(in_channels[i] * width),
                        out_channels=int(256 * width),
                        ksize=1,
                        stride=1,
                        act=act,
                    ),
            )
            self.stems_2.append(
                RepVGGBlock(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                ),
            )
            self.stems_3.append(
                RepVGGBlock(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                ),
            )
            self.cls_convs.append(
                    Conv(
                        in_channels=int(256 * width * 2),
                        out_channels=int(256 * width * 2),
                        ksize=1,
                        stride=1,
                        act=act,
                    )
            )
            self.reg_convs.append(
                    Conv(
                        in_channels=int(256 * width * 2),
                        out_channels=int(256 * width * 2),
                        ksize=1,
                        stride=1,
                        act=act,
                    )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width * 2),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width * 2),
                    out_channels=4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width * 2),
                    out_channels=self.n_anchors * 1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none", loss_type="giou")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            x_1 = self.stems_2[k](x)
            x_2 = self.stems_3[k](x_1)
            x = torch.cat((x_1, x_2), dim=1)
            cls_x = x
            reg_x = x
            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid( # 得到每个格子的输出和左上角的坐标
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])  # 得到x坐标
                y_shifts.append(grid[:, :, 1]) # 得到y坐标
                expanded_strides.append(  # 得到每个格子的步长
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype) # (1,1,h,w,2) 最后一维的2标最上角坐标
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape( # 将第二维调到最后
            batch_size, self.n_anchors * hsize * wsize, -1  # 不知道具体大小写-1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride # 最后一维前两个加左上角坐标乘步长预测左上角坐标
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride # 预测长和宽
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects  batchsize每个图片中有多少标注

        total_num_anchors = outputs.shape[1]  # 一共有多少个框
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all] 每个框的x坐标
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all] 每个框的y坐标
        expanded_strides = torch.cat(expanded_strides, 1)  # 步长
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        p_pos_weights = []
        p_neg_weights = []
        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])  # GT第batch_idx个框
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]  # 每张图片的bbox
                gt_classes = labels[batch_idx, :num_gt, 0]  # 真实分类
                bboxes_preds_per_image = bbox_preds[batch_idx]  # 每张图片的预测框
                if self.use_l1:
                    try:
                        (
                            gt_matched_classes,
                            fg_mask,
                            pred_ious_this_matching,
                            matched_gt_inds,
                            num_fg_img,
                            p_pos,
                            p_neg
                        ) = self.a_y_get_assignments(  # noqa
                            batch_idx,
                            num_gt,
                            total_num_anchors,
                            gt_bboxes_per_image,
                            gt_classes,
                            bboxes_preds_per_image,
                            expanded_strides,
                            x_shifts,
                            y_shifts,
                            cls_preds,
                            bbox_preds,
                            obj_preds,
                            labels,
                            imgs,
                        )
                    except RuntimeError as e:
                        # TODO: the string might change, consider a better way
                        if "CUDA out of memory. " not in str(e):
                            raise  # RuntimeError might not caused by CUDA OOM

                        logger.error(
                            "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                               CPU mode is applied in this batch. If you want to avoid this issue, \
                               try to reduce the batch size or image size."
                        )
                        torch.cuda.empty_cache()
                        (
                            gt_matched_classes,
                            fg_mask,
                            pred_ious_this_matching,
                            matched_gt_inds,
                            num_fg_img,
                            p_pos,
                            p_neg
                        ) = self.a_y_get_assignments(  # noqa
                            batch_idx,
                            num_gt,
                            total_num_anchors,
                            gt_bboxes_per_image,
                            gt_classes,
                            bboxes_preds_per_image,
                            expanded_strides,
                            x_shifts,
                            y_shifts,
                            cls_preds,
                            bbox_preds,
                            obj_preds,
                            labels,
                            imgs,
                            "cpu",
                        )
                else:
                    try:
                        (
                            gt_matched_classes,
                            fg_mask,
                            pred_ious_this_matching,
                            matched_gt_inds,
                            num_fg_img,
                        ) = self.get_assignments(  # noqa
                            batch_idx,
                            num_gt,
                            total_num_anchors,
                            gt_bboxes_per_image,
                            gt_classes,
                            bboxes_preds_per_image,
                            expanded_strides,
                            x_shifts,
                            y_shifts,
                            cls_preds,
                            bbox_preds,
                            obj_preds,
                            labels,
                            imgs,
                        )
                    except RuntimeError as e:
                        # TODO: the string might change, consider a better way
                        if "CUDA out of memory. " not in str(e):
                            raise  # RuntimeError might not caused by CUDA OOM

                        logger.error(
                            "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                               CPU mode is applied in this batch. If you want to avoid this issue, \
                               try to reduce the batch size or image size."
                        )
                        torch.cuda.empty_cache()
                        (
                            gt_matched_classes,
                            fg_mask,
                            pred_ious_this_matching,
                            matched_gt_inds,
                            num_fg_img,
                        ) = self.get_assignments(  # noqa
                            batch_idx,
                            num_gt,
                            total_num_anchors,
                            gt_bboxes_per_image,
                            gt_classes,
                            bboxes_preds_per_image,
                            expanded_strides,
                            x_shifts,
                            y_shifts,
                            cls_preds,
                            bbox_preds,
                            obj_preds,
                            labels,
                            imgs,
                            "cpu",
                        )
                torch.cuda.empty_cache()
                num_fg += num_fg_img

                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )
                    p_pos_weights.append(p_pos)
                    p_neg_weights.append(p_neg)
                    cls_target = F.one_hot(  # 正样本对应类别的iou
                        gt_matched_classes.to(torch.int64), self.num_classes
                    )
                else:
                    cls_target = F.one_hot(  # 正样本对应类别的iou
                        gt_matched_classes.to(torch.int64), self.num_classes
                    ) * pred_ious_this_matching.unsqueeze(-1)
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
            else:
                obj_targets.append(obj_target.to(dtype))

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        if self.use_l1:
            p_pos_weights = torch.cat(p_pos_weights, 0)
            p_neg_weights = torch.cat(p_neg_weights, 0)
            l1_targets = torch.cat(l1_targets, 0)
        else:
            obj_targets = torch.cat(obj_targets, 0)

        fg_masks = torch.cat(fg_masks, 0)


        num_fg = max(num_fg, 1)
        if self.use_l1:
            pos_af = p_pos_weights.sum()
            neg_af = (1 - p_neg_weights).sum()
            loss_iou = (
                               self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets) * p_pos_weights
                       ).sum() / pos_af
            with torch.cuda.amp.autocast(enabled=False):
                cls_preds = (
                        cls_preds.float().view(-1, self.num_classes).sigmoid()
                        * obj_preds.float().view(-1, 1).sigmoid().repeat(1, self.num_classes)
                )
                cls_weight = (p_pos_weights.unsqueeze(-1).repeat(1, self.num_classes) * cls_targets +
                              cls_preds[fg_masks].pow(2) * (1.0 - cls_targets)).to(dtype)
                cls_loss_p1 = (F.binary_cross_entropy(
                    cls_preds[fg_masks], cls_targets.float(), reduction="none"
                ) * cls_weight).sum() / pos_af
                neg = (cls_preds[fg_masks] * cls_targets).sum(-1)
                cls_loss_p2 = (F.binary_cross_entropy(
                    neg, torch.zeros_like(neg), reduction="none"
                ) * p_neg_weights).sum() / neg_af
                loss_obj = (F.binary_cross_entropy(
                    cls_preds[~fg_masks], torch.zeros_like(cls_preds[~fg_masks]), reduction="none"
                ) * cls_preds[~fg_masks].pow(2)).sum() / num_fg
            loss_cls = cls_loss_p1 + cls_loss_p2
            loss_l1 = (
                    (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum(-1) * p_pos_weights
                      ).sum() / pos_af
        else:
            loss_iou = (
                           self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                       ).sum() / num_fg
            loss_obj = (
                           self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                       ).sum() / num_fg
            loss_cls = (
                           self.bcewithlog_loss(  # 多标签分类损失
                               cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                           )
                       ).sum() / num_fg
            loss_l1 = 0.0

        reg_weight = 5.0
        fol_weight = 1.0
        loss_cls_align_weight = 1.0
        loss = reg_weight * loss_iou + loss_cls_align_weight * loss_cls + fol_weight * loss_obj + loss_l1


        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def normalize(self, max_, min_, inputs_v):
        k = (max_ - min_) / (inputs_v.max() - inputs_v.min() + 1e-12)
        return min_ + k * (inputs_v - inputs_v.min())

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def a_y_get_assignments(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )
        # 将get_in_boxes_info并集bool作为掩码提取为True的部分
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)  # tensor:(gt_classes,num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid() * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
        _, num_in_boxes_and_center, _ = cls_preds_.shape
        s_la = (cls_preds_ * gt_cls_per_image).sum(-1)
        s_other = (cls_preds_ * (1.0 - gt_cls_per_image)).max(-1).values
        p_cls = (pair_wise_ious * s_la + (1.0 - pair_wise_ious) * (1-s_other).pow(2))
        p_pos = p_cls * pair_wise_ious.pow(5)
        del cls_preds_
        p_pos[~is_in_boxes_and_center] = 0.0
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            pos,
            neg
        ) = self.a_y_dynamic_k_matching(p_pos, pair_wise_ious, gt_classes, num_gt, fg_mask, p_cls, is_in_boxes_and_center)
        del p_pos, pair_wise_ious

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
            pos,
            neg
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image  # 真实图像左上角坐标
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)    # 在外围加一个维度，方便计算
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )                                  # GT原图中心点坐标
        #计算真实框的四边
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])  #  中心点- 一半宽 = 左上角x  tensor:(3)
            .unsqueeze(1)    # tensor:(3,1)
            .repeat(1, total_num_anchors)  # tensor:(3,8400)
        )
        gt_bboxes_per_image_r = (                    # 右下角x
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (                    # 最上角y
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (                    # 右下角y
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        #判断8400个框的中心点有那些在真实框内
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
            # bool类型，若bbox_deltas的4个值中最小的小于零返回false
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
             #格子只要位于一个GT中便置为True
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        #取真实框中心为中心，格子边长5倍为边长的正方形
        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        # 判断8400个框的中心点是否在正方形内
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers 在真实框和在正方形取并集和交集
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def a_y_dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask, p_cls, is_in_boxes_and_center):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        pos = torch.zeros_like(cost)
        neg = torch.zeros_like(cost)
        for gt_idx in range(num_gt):
            ious_in_boxes_and_center_matrix = pair_wise_ious[gt_idx][is_in_boxes_and_center[gt_idx]]
            n_candidate_k = min(10, ious_in_boxes_and_center_matrix.size(0))
            topk_ious, _ = torch.topk(ious_in_boxes_and_center_matrix, n_candidate_k, dim=0)
            dynamic_k = torch.clamp(topk_ious.sum().int(), min=1)
            per_pos = cost[gt_idx]
            _, pos_idx = torch.topk(
                per_pos, k=dynamic_k, largest=True  # 取cost大小前dynamic_ks的anchor作为正样本
            )
            p_pos = per_pos[pos_idx]
            per_gt_cls = p_cls[gt_idx][pos_idx]
            matching_matrix[gt_idx][pos_idx] = 1  # 将所取位置在矩阵中变为1
            per_gt_iou_match = pair_wise_ious[gt_idx][pos_idx]
            # p_neg = (1 - per_gt_iou_match.pow(2)) * (1 - p_cls[gt_idx][pos_idx].pow(2))
            p_pos = (torch.exp(5 * p_pos) * p_pos * per_gt_iou_match)
            # p_pos = (p_pos / (p_pos.max() + 1e-16)) * per_gt_iou_match.max()
            p_pos = self.normalize(per_gt_iou_match.max(), per_gt_iou_match.min(), p_pos)
            pos[gt_idx][pos_idx] = p_pos
            p_neg_weight = (1 - per_gt_iou_match).pow(2) * (1 - per_gt_cls)
            max_ = (1 - per_gt_iou_match).max()
            min_ = (1 - per_gt_iou_match).min()
            p_neg_weight = self.normalize(max_, min_, p_neg_weight)
            # neg[gt_idx][pos_idx] = self.normalize(1 - per_gt_iou_match.min(), 1 - per_gt_iou_match.max(), p_neg)
            neg[gt_idx][pos_idx] = p_neg_weight
        del topk_ious, pos_idx, p_pos  # 释放内存

        anchor_matching_gt = matching_matrix.sum(0)  # 防止一个正样本匹配多个GT
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.max(cost[:, anchor_matching_gt > 1], dim=0)  # 取cost小的那个为自己对应的GT
            matching_matrix[:, anchor_matching_gt > 1] *= 0  # 大于一的全置为0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1  # cost最小的置为1
        fg_mask_inboxes = matching_matrix.sum(0) > 0  # 样本选择区间哪些被选择成正样本
        num_fg = fg_mask_inboxes.sum().item()  # 正样本个数
        pos = pos * matching_matrix
        neg = neg * matching_matrix
        fg_mask[fg_mask.clone()] = fg_mask_inboxes  # 全部点中哪些是正样本
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)  # 正样本对应GT 拿出matching_matrix中fg_mask_inboxes位置的数
        gt_matched_classes = gt_classes[matched_gt_inds]  # 正样本对应真实框类别
        pos = pos[:, fg_mask_inboxes].max(0).values
        neg = neg[:, fg_mask_inboxes].max(0).values
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[  # 每个正样本与真实框对应的IOU
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, pos, neg


    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )
        #将get_in_boxes_info并集bool作为掩码提取为True的部分
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)  # tensor:(gt_classes,num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )


    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)  # 排序，从大到小
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # 对应真实框所要选取的正样本个数
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False  #取cost大小前dynamic_ks的anchor作为正样本
            )
            matching_matrix[gt_idx][pos_idx] = 1   # 将所取位置在矩阵中变为1

        del topk_ious, dynamic_ks, pos_idx  # 释放内存

        anchor_matching_gt = matching_matrix.sum(0)  # 防止一个正样本匹配多个GT
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)  # 取cost小的那个为自己对应的GT
            matching_matrix[:, anchor_matching_gt > 1] *= 0   # 大于一的全置为0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1  # cost最小的置为1
        fg_mask_inboxes = matching_matrix.sum(0) > 0  # 样本选择区间哪些被选择成正样本
        num_fg = fg_mask_inboxes.sum().item()  # 正样本个数

        fg_mask[fg_mask.clone()] = fg_mask_inboxes  # 全部点中哪些是正样本

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0) # 正样本对应GT 拿出matching_matrix中fg_mask_inboxes位置的数
        gt_matched_classes = gt_classes[matched_gt_inds] # 正样本对应真实框类别

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[  # 每个正样本与真实框对应的IOU
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
