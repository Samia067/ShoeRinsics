import numpy as np
import os
from code.dataset.util import read_image, image_to_channels
from code.util.misc import get_invalid_tensor

class ValDataset(object):

    def __init__(self, input_dir, h=128, w=128):
        self.input_dir = input_dir
        self.patch_height = h
        self.patch_width = w

        image_dir = os.path.join(self.input_dir, 'image')
        mask_dir = os.path.join(self.input_dir, 'mask')
        print_dir = os.path.join(self.input_dir, 'print')
        self.GT_print_exists = os.path.exists(print_dir)
        albedo_dir = os.path.join(self.input_dir, 'albedo_pred_GT')
        self.GT_albedo_exists = os.path.exists(albedo_dir)

        self.image_file_names = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])

        self.image_files = np.empty(len(self.image_file_names), dtype=object) # []
        self.mask_files = np.empty(len(self.image_file_names), dtype=object)
        if self.GT_print_exists:
            self.print_files = np.empty(len(self.image_file_names), dtype=object)
        if self.GT_albedo_exists:
            self.albedo_files = np.empty(len(self.image_file_names), dtype=object)
        for i, file in enumerate(self.image_file_names):
            self.image_files[i] = os.path.join(image_dir, file)
            self.mask_files[i] = os.path.join(mask_dir, file)   # [:-4] + '.png'
            if self.GT_print_exists:
                self.print_files[i] = os.path.join(print_dir, file) # [:-4] + '.png'
            if self.GT_albedo_exists:
                self.albedo_files[i] = os.path.join(albedo_dir, file)   # [:-4] + '.png'

        print("Total shoes: ", len(self.image_files))

        # determine values to pad the full images with
        sample_image = read_image(self.image_files[0])
        shape = sample_image.shape
        h = self.patch_height
        while (h < shape[0]):
            h = h + self.patch_height
        w = self.patch_width
        while (w < shape[1]):
            w = w + self.patch_width

        self.pad_h_before = (h-shape[0]) // 2
        self.pad_h_after = (h-shape[0]) - self.pad_h_before
        self.pad_w_before = (w-shape[1]) // 2
        self.pad_w_after = (w-shape[1]) - self.pad_w_before



    def __getitem__(self, index):

        index = index % len(self.image_files)

        image = read_image(self.image_files[index])
        mask = read_image(self.mask_files[index], is_mask=True).astype(np.float)
        image = np.pad(image, ((self.pad_h_before, self.pad_h_after), (self.pad_w_before, self.pad_w_after), (0,0)), mode='edge')
        mask = np.pad(mask, ((self.pad_h_before, self.pad_h_after), (self.pad_w_before, self.pad_w_after), (0, 0)), mode='edge')

        # move the channel to the first dimension for training
        mask = np.round(mask).astype(bool)
        mask3d_inverted = ~mask.repeat(3, axis=2)
        image[mask3d_inverted] = 1
        image, mask = [image_to_channels(item) for item in [image, mask]]

        # read ground-truth print
        if self.GT_print_exists:
            print_ = read_image(self.print_files[index])
            print_ = np.pad(print_, ((self.pad_h_before, self.pad_h_after), (self.pad_w_before, self.pad_w_after), (0,0)), mode='edge')#, constant_values=1)
            print_ = image_to_channels(print_)
            print_ = print_[0:1, ...].astype(np.bool)
        else:
            print_ = get_invalid_tensor(tensor=False)

        # read albedo
        if self.GT_albedo_exists:
            albedo = read_image(self.albedo_files[index])[2:-3, ...]
            albedo = np.pad(albedo, ((self.pad_h_before, self.pad_h_after), (self.pad_w_before, self.pad_w_after), (0,0)), mode='edge')#, constant_values=1)
            albedo = image_to_channels(albedo)
        else:
            albedo = get_invalid_tensor(tensor=False)

        return image, mask[0:1, ...].astype(np.bool), print_, albedo, self.image_file_names[index], \
               self.pad_h_before, self.pad_h_after, self.pad_w_before, self.pad_w_after

    def __len__(self):
        return len(self.image_files)
