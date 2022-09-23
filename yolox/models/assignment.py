import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, meshgrid

class assignment(nn.Module):
    def __init__(self, num_classes):
        super(assignment, self).__init__()
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(
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
        obj_preds,
        use_simOTA=True,
        mode="gpu",):
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
        if use_simOTA:
            pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            if use_simOTA:
                pair_wise_cls_loss = F.binary_cross_entropy(
                    cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
                ).sum(-1)
                del cls_preds_
        if use_simOTA:
            T_scores = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
            )
            p_cls = None
        else:
            s_la = (cls_preds_ * gt_cls_per_image).sum(-1)
            s_other = (cls_preds_ * (1.0 - gt_cls_per_image)).max(-1).values
            p_cls = (pair_wise_ious * s_la + (1.0 - pair_wise_ious) * (1 - s_other).pow(2))
            T_scores = p_cls * pair_wise_ious.pow(5)
            del cls_preds_
            T_scores[~is_in_boxes_and_center] = 0.0
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            pos,
            neg
        ) = self.dynamic_k_matching(T_scores, use_simOTA, pair_wise_ious, gt_classes, num_gt, fg_mask, is_in_boxes_and_center, p_cls)
        del pair_wise_ious, T_scores
        if use_simOTA:
            del pair_wise_ious_loss, pair_wise_cls_loss

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

    def dynamic_k_matching(self, T_scores, use_simOTA, pair_wise_ious, gt_classes, num_gt, fg_mask, is_in_boxes_and_center, p_cls=None):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(T_scores, dtype=torch.uint8)
        if use_simOTA:
            ious_in_boxes_matrix = pair_wise_ious
            n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
            topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)  # 排序，从大到小
            dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # 对应真实框所要选取的正样本个数
            pos = None
            neg = None
        else:
            pos = torch.zeros_like(T_scores)
            neg = torch.zeros_like(T_scores)
        for gt_idx in range(num_gt):
            if not use_simOTA:
                ious_in_boxes_and_center_matrix = pair_wise_ious[gt_idx][is_in_boxes_and_center[gt_idx]]
                n_candidate_k = min(10, ious_in_boxes_and_center_matrix.size(0))
                topk_ious, _ = torch.topk(ious_in_boxes_and_center_matrix, n_candidate_k, dim=0)
                dynamic_k = torch.clamp(topk_ious.sum().int(), min=1)
            else:
                dynamic_k = dynamic_ks[gt_idx]
            per_pos = T_scores[gt_idx]
            _, pos_idx = torch.topk(
                per_pos, k=dynamic_k, largest=not use_simOTA  # 取cost大小前dynamic_ks的anchor作为正样本
            )
            matching_matrix[gt_idx][pos_idx] = 1  # 将所取位置在矩阵中变为1
            if not use_simOTA:
                p_pos = per_pos[pos_idx]
                per_gt_cls = p_cls[gt_idx][pos_idx]
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

        del topk_ious, pos_idx  # 释放内存
        if use_simOTA:
            del dynamic_ks

        anchor_matching_gt = matching_matrix.sum(0)  # 防止一个正样本匹配多个GT
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.max(T_scores[:, anchor_matching_gt > 1], dim=0)  # 取cost小的那个为自己对应的GT
            matching_matrix[:, anchor_matching_gt > 1] *= 0  # 大于一的全置为0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1  # cost最小的置为1
        fg_mask_inboxes = matching_matrix.sum(0) > 0  # 样本选择区间哪些被选择成正样本
        num_fg = fg_mask_inboxes.sum().item()  # 正样本个数
        if not use_simOTA:
            pos = pos * matching_matrix
            neg = neg * matching_matrix
        fg_mask[fg_mask.clone()] = fg_mask_inboxes  # 全部点中哪些是正样本
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)  # 正样本对应GT 拿出matching_matrix中fg_mask_inboxes位置的数
        gt_matched_classes = gt_classes[matched_gt_inds]  # 正样本对应真实框类别
        if not use_simOTA:
            pos = pos[:, fg_mask_inboxes].max(0).values
            neg = neg[:, fg_mask_inboxes].max(0).values
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[  # 每个正样本与真实框对应的IOU
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, pos, neg

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

    def normalize(self, max_, min_, inputs_v):
        k = (max_ - min_) / (inputs_v.max() - inputs_v.min() + 1e-12)
        return min_ + k * (inputs_v - inputs_v.min())