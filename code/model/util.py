import torch
from code.util.misc import gaussian_kernel, eps
import numpy as np


'''
Returns function to concatenate tensors.
Used in skip layers to join encoder and decoder layers. '''
def join(ind):
    return lambda x, y: torch.cat((x, y), ind)

'''Normalize tensor to unit vectors'''
def normalize(normals):
    magnitude = torch.pow(normals, 2).sum(1)
    magnitude = magnitude.sqrt().unsqueeze(1).repeat(1, 3, 1, 1)
    normed = normals / (magnitude + 1e-6)
    return normed

"""Get the value (or shift) that needs to be added to the current patch to match the previous patch."""
def get_shift(prev_patch, current_patch, mask):
    if mask.shape[1] == 1 and prev_patch.shape[1] == 3:
        mask = mask.repeat(1, 3, 1, 1)

    diff = current_patch - prev_patch
    valid_values = torch.sort(diff[mask]).values
    valid_values = valid_values[len(valid_values) // 10:-len(valid_values) // 10]
    if len(valid_values) == 0:
        shift = 0
    else:
        shift = torch.mean(valid_values)
    return - shift


class Stich():
    def __init__(self, shape, w, h, device, min_val, max_val, get_patch):
        self.shape = shape
        self.w = w
        self.h = h
        self.allowed_min_val = min_val
        self.allowed_max_val = max_val
        self.device = device
        self.image = torch.zeros(self.shape).to(self.device)
        self.max_adjusted = torch.zeros(self.shape).to(self.device)
        self.min_adjusted = torch.ones(self.shape).to(self.device)
        self.prev_col_overlap = None
        self.shift = 0
        self.get_patch = get_patch

    def new_col(self):
        self.col_shape = (self.shape[0], self.shape[1], self.shape[2], self.w)
        self.col = torch.zeros(self.col_shape).to(self.device)
        self.max_adjusted_col = torch.zeros(self.col_shape).to(self.device)
        self.min_adjusted_col = torch.ones(self.col_shape).to(self.device)
        self.prev_patch_overlap = None

    def reset_prev_patch_overlap(self):
        self.prev_patch_overlap = None

    def patch_overlap(self):
        return self.prev_patch_overlap is not None

    def set_overlap_mask(self, overlap_mask):
        self.overlap_mask = overlap_mask

    def set_overlap_mask_col(self, overlap_mask_col):
        self.overlap_mask_col = overlap_mask_col

    def shift_patch(self, patch):
        self.prev_patch_overlap = self.prev_patch_overlap * self.overlap_mask.float()
        self.current_patch_overlap = patch[..., :self.w // 2, :] * self.overlap_mask.float()
        return patch + get_shift(self.prev_patch_overlap, self.current_patch_overlap, self.overlap_mask)

    def update_prev_patch(self, patch, mask):
        self.prev_patch_overlap = patch[..., self.w // 2:, :] * mask[..., self.w // 2:, :].float()
        # mask[:, 0:1, 128 // 2:, :].float()

    def update_columns(self, patch, index):
        self.col[:, :, index:index + self.w, :] += patch

    def update_col_error_helpers(self, patch, index, m, eps):
        patch[m.repeat(1, self.shape[1], 1, 1) < eps] = 0
        self.max_adjusted_col[:, :, index:index + self.w, :] = torch.max(self.max_adjusted_col[:, :, index:index + self.w, :], patch)
        patch[m.repeat(1, self.shape[1], 1, 1) < eps] = 1
        self.min_adjusted_col[:, :, index:index + self.w, :] = torch.min(self.min_adjusted_col[:, :, index:index + self.w, :], patch)

    def update_error_helpers(self, index):
        self.max_adjusted[:, :, :, index:index + self.w] = torch.max(self.max_adjusted[:, :, :, index:index + self.w], self.max_adjusted_col + self.shift)
        self.min_adjusted[:, :, :, index:index + self.w] = torch.min(self.min_adjusted[:, :, :, index:index + self.w], self.min_adjusted_col + self.shift)

    def update_prev_col_overlap(self):
        self.prev_col_overlap = self.col_without_mul[..., self.w // 2:]

    def update_image_col(self, index, mask_sum_col):
        self.image[:, :, :, index:index + self.w] += self.col_without_mul * mask_sum_col * (mask_sum_col > 0).float()

    def update_image(self, mask_sum):
        self.image[(mask_sum == 0).repeat((1, self.shape[1], 1, 1))] = 1
        mask_sum = mask_sum.clone()
        mask_sum[mask_sum == 0] = 1
        self.image = self.image / mask_sum
        current_min_val = torch.min(self.image)
        if current_min_val < self.allowed_min_val:
            self.image += -current_min_val + self.allowed_min_val
        current_max_val = torch.max(self.image)
        if current_max_val > self.allowed_max_val:
            self.image = self.image - self.allowed_min_val
            curr_max = torch.max(self.image)
            range = self.allowed_max_val - self.allowed_min_val
            self.image = self.image * range / curr_max + self.allowed_min_val

    def set_col_without_mul(self, mask_for_division):
        self.col_without_mul = self.col / mask_for_division

    def col_overlap(self):
        return self.prev_col_overlap is not None

    def shift_col(self):
        self.prev_col_overlap = self.prev_col_overlap * self.overlap_mask_col.float()
        # self.current_col_overlap = (col[..., :self.w // 2] / mask_for_division[..., :self.w // 2]) * self.overlap_mask_col.float()
        self.current_col_overlap = self.col_without_mul[..., :self.w // 2] * self.overlap_mask_col.float()
        shift = get_shift(self.prev_col_overlap, self.current_col_overlap, self.overlap_mask_col)
        self.col_without_mul += shift

    def prepare_error(self, mask):
        self.error = (self.max_adjusted - self.min_adjusted) * mask.float()

    def stich(self, i_list, j_list, patches, mask, patch_interval, device, mean_shift=True):

        mask_sum = torch.zeros((self.shape[0], 1, self.shape[2], self.shape[3])).to(device)
        mask_blend = gaussian_kernel(kernlen=self.h, nsig=2)
        mask_blend = mask_blend - np.max(mask_blend[:, 0])
        mask_blend[mask_blend < 0] = 0
        mask_blend = torch.tensor(mask_blend / np.max(mask_blend)).float().to(device)
        mask_blend_edge = mask_blend[:, mask_blend.shape[1] // 2].unsqueeze(1).repeat((1, 64))
        mask_blend_left = mask_blend.clone()
        mask_blend_left[:, :mask_blend.shape[1] // 2] = mask_blend_edge
        mask_blend_right = mask_blend.clone()
        mask_blend_right[:, mask_blend.shape[1] // 2:] = mask_blend_edge
        mask_valid_max = mask.shape[1] * self.h * self.w
        overlap_mask_col = torch.ones((self.shape[0], 1, self.shape[2], self.w // 2)).to(device) < 0

        index = -1
        for i_ind, i in enumerate(i_list):
            self.new_col()

            mask_sum_col = torch.zeros((self.shape[0], 1, self.shape[2], self.w)).to(device)
            overlap_mask = torch.ones((self.shape[0], 1, self.h // 2, self.w)).to(device) < 0

            for j_ind, j in enumerate(j_list):
                index += 1
                mask_patch = self.get_patch(mask, i, j)
                mask_valid = torch.sum(mask_patch.reshape((mask_patch.shape[0], -1)), dim=1, dtype=float)
                mask_fraction = (mask_valid / float(mask_valid_max)).float()
                if torch.sum(mask_fraction) <= 0:
                    self.reset_prev_patch_overlap()
                    continue

                mask_fraction_reshaped = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(mask_fraction, 1), 1), 1).float()
                if i == 0:
                    m = (mask_fraction_reshaped * mask_blend_left * mask_patch.float()).float()
                elif i >= int(self.shape[3] - self.w) + 1 - int(self.w * patch_interval):
                    m = (mask_fraction_reshaped * mask_blend_right * mask_patch.float()).float()
                else:
                    m = (mask_fraction_reshaped * mask_blend * mask_patch.float()).float()

                # if predictions are going to be masked out because of blending, skip this iteration
                if torch.sum(m) < eps:
                    self.reset_prev_patch_overlap()
                    continue

                patch = patches[index:index + 1].to(device)
                overlap_mask = overlap_mask * (m[..., :128 // 2, :] > 0)
                self.set_overlap_mask(overlap_mask.repeat((1, self.shape[1], 1, 1)))

                if mean_shift and self.patch_overlap() and torch.sum(overlap_mask) != 0:
                    patch = self.shift_patch(patch)

                overlap_mask = mask_patch[..., 128 // 2:, :].bool() * (m[..., 128 // 2:, :] > 0)

                # update prev patches for next iteration
                self.update_prev_patch(patch, mask_patch)

                # update column values
                self.update_columns(patch * m, j)

                # update max and min values. These are used to calculate error
                self.update_col_error_helpers(patch, j, m, eps)
                mask_sum_col[:, :, j:j + 128, :] += m[:, 0:1, ...]


            overlap_mask_col = overlap_mask_col * (mask_sum_col[..., :128 // 2] > 0)
            mask_for_division = mask_sum_col.clone()
            mask_for_division[mask_for_division < eps] = 1
            self.set_overlap_mask_col(overlap_mask_col.repeat(1, self.shape[1], 1, 1))
            self.set_col_without_mul(mask_for_division)

            if mean_shift and self.col_overlap():
                self.shift_col()

            # update previous overlap values
            overlap_mask_col = mask_sum_col[..., 128 // 2:] > 0
            mask_sum[:, :, :, i:i + 128] += mask_sum_col

            self.update_prev_col_overlap()
            self.update_image_col(i, mask_sum_col)
            self.update_error_helpers(i)

        self.update_image(mask_sum)
        self.prepare_error(mask)

