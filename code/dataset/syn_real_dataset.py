import os.path
import numpy as np

from .real_dataset import RealDataset
from .syn_dataset import SynDataset


class SynRealDataset:

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        preload_type_count = 60
        get_count_per_load = 400
        self.dataset_size = 5000

        self.dir_A = os.path.join(opt.dataroot, opt.syn_train_dataset_dir, opt.phase)
        self.syn_shoe_dataset = SynDataset(self.dir_A, dataset_size=self.dataset_size, preload_type_count=preload_type_count,
                                                 get_count_per_load=get_count_per_load)


        self.dir_B = os.path.join(opt.dataroot, opt.real_train_dataset_dir, opt.phase)
        self.real_shoe_dataset = RealDataset(self.dir_B, dataset_size=self.dataset_size, preload_type_count=preload_type_count,
                                             get_count_per_load=get_count_per_load)



    def __getitem__(self, index):

        syn_items = self.syn_shoe_dataset.__getitem__(index)
        render, syn_albedo, depth, normal, light, light_type, syn_mask, syn_name = syn_items


        real_items = self.real_shoe_dataset.__getitem__(index)
        real_image, real_albedo, real_mask, real_name = real_items

        return render, syn_albedo, depth, normal, light, light_type, syn_mask, syn_name, real_image, real_albedo, real_mask, real_name


    def __len__(self):
        return self.dataset_size


