import numpy as np
import os
import torch
from .util import read_image, is_RGB, image_to_channels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RealDataset(object):

    def __init__(self, input_dir, w=128, h=128, dataset_size=2000, preload_type_count=200, get_count_per_load=3000, full=False):
        self.input_dir = input_dir
        self.patch_width = w
        self.patch_height = h
        self.dataset_size = dataset_size
        self.full = full

        image_dir = os.path.join(self.input_dir, 'image')
        albedo_dir = os.path.join(self.input_dir, 'albedo')
        mask_dir = os.path.join(self.input_dir, 'mask')

        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])

        self.image_files = []
        self.albedo_files = []
        self.mask_files = []
        shoe_type_index = -1
        shoe_type = ''
        print("Processing real shoe dataset")
        for i, file in enumerate(image_files):
            filename_parts = file.split("_")
            new_shoe_type = filename_parts[0]

            # the sorted files are ordered such that all shoes of the same type appear contiguously
            # check if we have reached a new processed_shoe type
            if new_shoe_type != shoe_type:
                shoe_type_index += 1
                shoe_type = new_shoe_type

                for item in [self.image_files, self.albedo_files, self.mask_files]:
                    item.append([])

            self.image_files[shoe_type_index].append(os.path.join(image_dir, file))
            self.albedo_files[shoe_type_index].append(os.path.join(albedo_dir, file))
            self.mask_files[shoe_type_index].append(os.path.join(mask_dir, file))

        print("Total shoe types: ", len(self.image_files))

        count = 0
        for shoe_type in self.image_files:
            count_in_type = len(shoe_type)
            combinations = count_in_type*(count_in_type - 1)/2
            count += combinations

        print("Total shoe pairs: ", count)

        self.preload_size = np.minimum(preload_type_count, len(image_files))
        self.get_count = 0
        self.get_count_per_load = get_count_per_load
        self.load()



    def append_images(self, sub_shoe_index, main_shoe_type_index, main_image_index):

        self.paths[sub_shoe_index].append(self.image_files[main_shoe_type_index][main_image_index])

        mask = read_image(self.mask_files[main_shoe_type_index][main_image_index], is_mask=True).astype(bool)
        mask3d_inverted = ~mask.repeat(3, axis=2)
        self.masks[sub_shoe_index].append(mask)

        image = read_image(self.image_files[main_shoe_type_index][main_image_index])
        image[mask3d_inverted] = 1
        self.images[sub_shoe_index].append(image)

        albedo = read_image(self.albedo_files[main_shoe_type_index][main_image_index])
        albedo[mask3d_inverted] = 1
        self.albedos[sub_shoe_index].append(albedo)




    """Load self.preload_type_count number of images for use in self.get_item"""
    def load(self):

        self.masks = []
        self.images = []
        self.albedos = []
        self.paths = []
        indices = []
        specific_shoe_indices = []
        for i in range(self.preload_size):
            main_shoe_type_index, main_image_index = self.get_image_index(all=True)

            if main_shoe_type_index not in indices:
                load_shoe_index = len(indices)
                indices.append(main_shoe_type_index)
                specific_shoe_indices.append([])
                specific_shoe_indices[load_shoe_index].append(main_image_index)

                for item in [self.images, self.albedos, self.masks, self.paths]:
                    item.append([])

                self.append_images(load_shoe_index, main_shoe_type_index, main_image_index)

            else:
                load_shoe_index = indices.index(main_shoe_type_index)
                if main_image_index not in specific_shoe_indices[load_shoe_index]:
                    self.append_images(load_shoe_index, main_shoe_type_index, main_image_index)
                    specific_shoe_indices[load_shoe_index].append(main_image_index)


        [self.images, self.albedos, self.masks] = \
            [item if item[0][0] is not None else None for item in [self.images, self.albedos, self.masks]]
        self.get_count = 0


    """
        Returns a patch of size self.patch_height * self.patch_width from image. 
        The upper left corner (with small x and y coordinates) of the image patch is specified by x and y.
        It works for both RGB and grayscale images
        """

    def get_patch(self, image, x, y):
        if image is None:
            return None
        if is_RGB(image):
            return image[y:y + self.patch_height, x:x + self.patch_width, :]
        else:
            return image[y:y + self.patch_height, x:x + self.patch_width]

    """
        Given a shoe_type and the indices of a pair of images for that shoe_type, return a random patch
        from the two images and the mask associated with the pair.
        """

    def get_random_patch(self, shoe_type_index, index, x=None, y=None):
        if x is None:
            # get maximum possible starting values for x and y coordinate of patch
            x_max = self.images[shoe_type_index][index].shape[1] - self.patch_width - 1
            y_max = self.images[shoe_type_index][index].shape[0] - self.patch_height - 1

            # get a random starting value for x and y coordinate of patch
            x = int(x_max * np.random.rand())
            y = int(y_max * np.random.rand())

        [image, albedo, mask] =  [self.get_patch(item[shoe_type_index][index], x, y) if item is not None else None
             for item in [self.images, self.albedos, self.masks]]

        return image, albedo, mask.astype(np.bool), x, y

    """
        Returns a random processed_shoe type and a random distinct pair of images for that processed_shoe type. 
        Returned values are indices used to access self.rendered_files, self.depth_files and self.albedo_file_paths
        """

    def get_image_index(self, all=False):
        if all:
            files = self.image_files
        else:
            files = self.images

        # get a random processed_shoe type
        shoe_type_index = int(np.random.rand() * len(files))

        # get a random pair of images of this shoe_type
        image_index = int(np.random.rand() * len(files[shoe_type_index]))


        return shoe_type_index, image_index

    def __getitem__(self, index):
        self.get_count += 1
        if self.get_count > self.get_count_per_load:
            self.load()

        repeat = True
        while (repeat):
            # generate a random processed_shoe type and a random pair of shoes of that processed_shoe type
            shoe_type_index, shoe_index = self.get_image_index()

            # get a random patch of the processed_shoe from each image in the pair
            if self.full:
                [image, albedo, mask] = [item[shoe_type_index][shoe_index] if item is not None else None
                                          for item in [self.images, self.albedos, self.masks]]
                mask = mask.astype(np.bool)
                repeat = False
            else:
                image, albedo, mask, x, y = self.get_random_patch(shoe_type_index, shoe_index)
                shoe_fraction = np.sum(mask) / mask.size
                if shoe_fraction >= 0.5:
                    repeat = False

        # move the channel to the first dimension for training
        [image, albedo, mask] = [image_to_channels(item) if item is not None else -1 for item in [image, albedo, mask]]
        name = self.paths[shoe_type_index][shoe_index].rsplit('/', 1)[1]

        return image, albedo, mask[0:1, ...].astype(np.bool), name

    def __len__(self):
        return self.dataset_size

