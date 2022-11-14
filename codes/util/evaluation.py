import torch, torch.nn as nn
import numpy as np
from codes.util.misc import valid_tensor


'''Calculate Intersection over Union (IoU) between print_pred and print_GT. IoU is only calculated in area specified by mask.'''
def iou(print_pred, print_GT, mask):
    intersection = (~print_pred * ~print_GT)
    union = (~print_pred + ~print_GT)
    intersection_sum = torch.sum(intersection[mask])
    union_sum = torch.sum(union[mask])
    return 100.0 * intersection_sum / union_sum


"""Given a depth map or an RGB image of a shoe, get an estimate for the print left on the ground"""
def get_print(t, mask, print_GT):
    if t.shape[0] > 1:
        prints = torch.ones(t.shape, dtype=torch.bool).to(t.device)

        for i in range(t.shape[0]):
            ind_prints = get_print(t[i:i+1, ...], mask[i:i+1, ...], print_GT[i:i+1, ...] if print_GT is not None else None)
            prints[i, ...] = ind_prints

        return prints

    if t.shape[1] == 3:
        tensor = 1 - t.mean(dim=1).unsqueeze(1).clamp(0, 1)
        kernel_size = 15
        thresh = 1
    else:
        tensor = t
        kernel_size = 45
        thresh = 1
    tensor = tensor - torch.min(tensor)

    kernel = (torch.ones((1, 1, kernel_size, kernel_size)) / (kernel_size * kernel_size)).to(t.device)
    depth_blur = nn.functional.conv2d(mask * tensor, kernel, padding=kernel_size//2)
    mask_blur = nn.functional.conv2d(mask.float(), kernel, padding=kernel_size//2)
    blur = depth_blur / mask_blur
    outside_shoe = ~mask.bool()
    blur[outside_shoe] = 0

    if t.shape[1] == 3 or not valid_tensor(print_GT):
        print_ = tensor < (blur * thresh)
        print_[outside_shoe] = 0
        print_ = ~print_
        depth_vals = torch.sort(tensor[mask])[0]
        high_depth_val = depth_vals[int(depth_vals.shape[0] * .95)]
        print_[tensor > high_depth_val] = True
        low_depth_val = depth_vals[int(depth_vals.shape[0] * .05)]
        print_[tensor < low_depth_val] = False
    else:
        thresh_min = 0.1
        thresh_max = 2
        int_over_uni = -1
        ious = [None] * 190
        for i, thresh in enumerate(np.arange(thresh_min, thresh_max, 0.01)):

            print_ = tensor < (blur * thresh)
            print_[outside_shoe] = 0
            print_ = ~print_
            new_int_over_uni = iou(print_, print_GT, mask)
            ious[i] = new_int_over_uni
            if new_int_over_uni > int_over_uni:
                best_thresh = thresh
                int_over_uni = new_int_over_uni

        print_ = tensor < (blur * best_thresh)
        print_[outside_shoe] = 0
        print_ = ~print_
        best_iou = iou(print_, print_GT, mask)

        # specify very high values as non contact surface
        depth_vals = tensor[mask]
        high_depth_val = torch.sort(depth_vals)[0][int(depth_vals.shape[0] * .95)]
        best_upper_depth_thresh = None
        for i, thresh in enumerate(np.arange(0.1, 1, 0.01)):
            new_iou = iou(print_ + (tensor > (high_depth_val * thresh)), print_GT, mask)
            if new_iou > best_iou:
                best_upper_depth_thresh = thresh
                best_iou = new_iou
        if best_upper_depth_thresh:
            print_[tensor > high_depth_val * best_upper_depth_thresh] = True

        # specify very low values as contact surface
        low_depth_val = torch.sort(depth_vals)[0][int(depth_vals.shape[0] * .05)]
        best_lower_depth_thresh = None
        for i, thresh in enumerate(np.arange(1, 30, 0.1)):
            new_iou = iou(print_ * (tensor > (low_depth_val * thresh)), print_GT, mask)
            if new_iou > best_iou:
                best_lower_depth_thresh = thresh
                best_iou = new_iou
        if best_lower_depth_thresh:
            print_[tensor < low_depth_val * best_lower_depth_thresh] = False

    return print_