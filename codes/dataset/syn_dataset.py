import numpy as np
import cv2
import os
import random
from collections import deque
from .util import read_image, image_to_channels, is_RGB

class SynDataset(object):

    def __init__(self, input_dir, w=128, h=128, dataset_size=2000, preload_type_count=200, get_count_per_load=3000,
                 max_shoe_per_type=17, full=False):

        self.input_dir = input_dir
        self.patch_width = w
        self.patch_height = h
        self.dataset_size = dataset_size
        self.full = full

        render_dir = os.path.join(self.input_dir, 'render')
        albedo_dir = os.path.join(self.input_dir, 'albedo')
        depth_dir = os.path.join(self.input_dir, 'depth')
        light_dir = os.path.join(self.input_dir, 'light')
        mask_dir = os.path.join(self.input_dir, 'mask')
        normal_dir = os.path.join(self.input_dir, 'normal')

        light_npy_files = sorted([f for f in os.listdir(light_dir) if f.endswith('.npy')])
        self.light_maps = np.array([np.load(os.path.join(light_dir, light)) for light in light_npy_files])[..., np.newaxis]
        # self.light_visuals = np.array([image_to_channels(cv2.imread(os.path.join(light_dir, light_npy_files[i].split(".")[0] + ".png"))) for i in
        #                                      range(len(light_npy_files))], dtype=np.float32)/255


        self.shoe_types = np.sort(np.array([f[:-4] for f in os.listdir(depth_dir) if f.endswith('.png')]))
        self.total_shoe_types = len(self.shoe_types)
        self.max_shoe_per_type = max_shoe_per_type
        self.shoe_type_length = len(self.shoe_types[0])
        self.depth_paths = [os.path.join(depth_dir, file[:self.shoe_type_length] + ".png") for file in self.shoe_types]
        self.mask_paths = [os.path.join(mask_dir, file[:self.shoe_type_length] + ".png") for file in self.shoe_types]
        self.normal_paths = [os.path.join(normal_dir, file[:self.shoe_type_length] + ".png") for file in self.shoe_types]

        renders = np.sort(np.array([f for f in os.listdir(render_dir) if f.endswith('.png')]))

        r_start = 0
        self.render_paths = []
        self.albedo_paths = []
        if self.full:
            self.map_index=deque()

        print("Processing syn shoe dataset")
        for shoe_type_index, d in enumerate(self.shoe_types):
            d_renders = []
            d_albedos = []

            loop_list = renders
            for i, r in enumerate(loop_list[r_start:]):
                if r.startswith(d):
                    if self.full:
                        self.map_index.append([shoe_type_index, i])
                    d_renders.append(os.path.join(render_dir, r))
                    d_albedos.append(os.path.join(albedo_dir, r))
                else:
                    r_start += i
                    break
            self.render_paths.append(d_renders)
            self.albedo_paths.append(d_albedos)

        self.light_index = np.zeros(shape=(len(self.render_paths), self.max_shoe_per_type), dtype=np.int)
        for i, shoe_type in enumerate(self.render_paths):
            for j, render_file in enumerate(shoe_type):
                self.light_index[i, j] = render_file.rsplit("/", 1)[-1][-6:-4]

        # sanity checks
        for shoe_types in self.albedo_paths:
            for albedo_file_path in shoe_types:
                if not os.path.exists(albedo_file_path):
                    raise FileNotFoundError(albedo_file_path + "not found!")


        # list statistics
        print("Total shoe types: ", len(self.render_paths))
        pair_count = 0
        count = 0
        files_to_iterate = self.render_paths
        self.min_shoe_per_type = np.inf
        for shoe_type in files_to_iterate:
            count_in_type = len(shoe_type)
            self.min_shoe_per_type = np.minimum(self.min_shoe_per_type, count_in_type)
            combinations = count_in_type * (count_in_type - 1) / 2
            pair_count += combinations
            count += count_in_type
        self.min_shoe_per_type = int(self.min_shoe_per_type)

        print("Total shoe pairs: ", int(pair_count))
        print("Total shoe count: ", count)
        self.data_count = count

        self.preload_type_count = preload_type_count
        self.get_count_per_load = get_count_per_load
        self.preload_shoes_per_type = np.minimum(5, self.min_shoe_per_type)

        self.get_count = 0
        self.shape = read_image(self.depth_paths[0], gray=True).shape[:-1]

        self.load()

    def get_depth_image(self, index):
        return read_image(self.depth_paths[index], gray=True)

    def get_normal_image(self, index):
        normal = read_image(self.normal_paths[index])
        normal = 2 * (normal - 0.5)
        return normal

    def get_mask_image(self, index):
        return read_image(self.mask_paths[index], is_mask=True).astype(np.bool)

    def get_path(self, type_index, shoe_index):
        return self.render_paths[type_index][shoe_index]

    def get_render_image(self, type_index, shoe_index):
        return read_image(self.render_paths[type_index][shoe_index])


    def get_albedo_image(self, type_index, shoe_index):
        return read_image(self.albedo_paths[type_index][shoe_index])



    """Load self.preload_type_count number of images for use in self.get_item"""
    def load(self):
        self.paths = np.empty((self.preload_type_count, self.preload_shoes_per_type), dtype=object)
        self.depths = np.zeros((self.preload_type_count, self.shape[0], self.shape[1], 1))
        self.normals = np.zeros((self.preload_type_count, self.shape[0], self.shape[1], 3))
        self.masks = np.zeros((self.preload_type_count, self.shape[0], self.shape[1], 1))
        self.renders = np.zeros((self.preload_type_count, self.preload_shoes_per_type, self.shape[0], self.shape[1], 3))
        self.albedos = np.zeros((self.preload_type_count, self.preload_shoes_per_type, self.shape[0], self.shape[1], 3))
        self.lights = np.zeros((self.preload_type_count, self.preload_shoes_per_type), np.int)

        selected_shoe_types_indices = random.sample(range(0, self.total_shoe_types), self.preload_type_count)
        for preload_type_index, type_index in enumerate(selected_shoe_types_indices):
            self.depths[preload_type_index, ...] = self.get_depth_image(type_index)
            self.normals[preload_type_index, ...] = self.get_normal_image(type_index)
            self.masks[preload_type_index, ...] = self.get_mask_image(type_index)

            selected_shoe_indices = random.sample(range(len(self.render_paths[type_index])), self.preload_shoes_per_type)
            for preload_shoe_index, shoe_index in enumerate(selected_shoe_indices):

                self.paths[preload_type_index, preload_shoe_index] = self.get_path(type_index, shoe_index)

                # load rendered image
                self.renders[preload_type_index, preload_shoe_index, ...] = self.get_render_image(type_index, shoe_index)

                # load albedo image
                self.albedos[preload_type_index, preload_shoe_index, ...] = self.get_albedo_image(type_index, shoe_index)

                # save light
                self.lights[preload_type_index, preload_shoe_index] = self.light_index[type_index][shoe_index]

        self.get_count = 0


    """
        Returns a patch of size self.patch_height * self.patch_width from image. 
        The upper left corner (with small x and y coordinates) of the image patch is specified by x and y.
        It works for both RGB and grayscale images
        """

    def get_patch(self, image, x, y):
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
            x_max = self.renders[shoe_type_index][index].shape[1] - self.patch_width - 1
            y_max = self.renders[shoe_type_index][index].shape[0] - self.patch_height - 1

            # get a random starting value for x and y coordinate of patch
            x = int(x_max * np.random.rand())
            y = int(y_max * np.random.rand())

        render = self.get_patch(self.renders[shoe_type_index][index], x, y)
        mask = self.get_patch(self.masks[shoe_type_index], x, y).astype(np.bool)
        albedo = self.get_patch(self.albedos[shoe_type_index][index], x, y)

        depth = self.get_patch(self.depths[shoe_type_index], x, y)
        normal = self.get_patch(self.normals[shoe_type_index], x, y)

        return render, albedo, depth, normal, self.light_maps[self.lights[shoe_type_index][index]], \
               self.lights[shoe_type_index][index], mask, x, y

    """
        Returns a random processed_shoe type and a random distinct pair of images for that processed_shoe type. 
        Returned values are indices used to access self.rendered_files, self.depth_files and self.albedo_paths
        """

    def get_random_image_index(self, all=False):
        if all:
            files = self.render_paths
        else:
            files = self.renders

        # get a random processed_shoe type
        if isinstance(files, list):
            shoe_type_count = len(files)
        else:
            shoe_type_count = files.shape[0]
        shoe_type_index = int(np.random.rand() * shoe_type_count)

        # get a random image index from this shoe_type
        if isinstance(files[shoe_type_index], list):
            file_count_of_shoe_type_index = len(files[shoe_type_index])
        else:
            file_count_of_shoe_type_index = files.shape[1]
        image_index = int(np.random.rand() * file_count_of_shoe_type_index)

        return shoe_type_index, image_index

    def __getitem__(self, index):
        if self.full:
            shoe_type_index, albedo_light_type_index = self.map_index[index]
            depth = self.get_depth_image(shoe_type_index)
            normal = self.get_normal_image(shoe_type_index)
            mask = self.get_mask_image(shoe_type_index)
            name = self.get_path(shoe_type_index, albedo_light_type_index)
            render = self.get_render_image(shoe_type_index, albedo_light_type_index)
            albedo = self.get_albedo_image(shoe_type_index, albedo_light_type_index)
            light_type = self.light_index[shoe_type_index][albedo_light_type_index]
            light = self.light_maps[light_type]
        else:
            self.get_count += 1
            if self.get_count > self.get_count_per_load:
                self.load()

            repeat = True
            while (repeat):
                # generate a random processed_shoe type and a random pair of shoes of that processed_shoe type
                shoe_type_index, image_index = self.get_random_image_index()
                name = self.paths[shoe_type_index][image_index].rsplit('/', 1)[1]

                # get a random patch of the processed_shoe from each image in the pair
                render, albedo, depth, normal, light, light_type, mask, x, y = \
                    self.get_random_patch(shoe_type_index, image_index)

                shoe_fraction = np.sum(mask) / mask.size
                if shoe_fraction >= 0.5:
                    repeat = False

        # move the channel to the first dimension for training
        render, albedo, light, depth, normal, mask = [image_to_channels(item) if len(item.shape) == 3 else item
                                                      for item in [render, albedo, light, depth, normal, mask]]

        mask = mask.astype(bool)

        return render, albedo, depth, normal, light, light_type, mask, name


    def __len__(self):
        if self.full:
            return self.data_count
        return self.dataset_size


# import shutil
# name_map = {}
# counter = 1
# prev_trace = ''
# for shoe_type in self.shoe_types:
#     cur_trace = shoe_type[:5]
#     if prev_trace == cur_trace:
#         counter += 1
#     else:
#         counter = 1
#     name_map[shoe_type] = cur_trace + '_' + str(counter).zfill(4)
#     prev_trace = cur_trace
# os.makedirs(os.path.join(depth_dir)+'_new')
# for key in name_map.keys():
#     shutil.copy(os.path.join(depth_dir, key) + '.png', os.path.join(depth_dir+'_new', name_map[key]) + '.png')
# os.makedirs(os.path.join(mask_dir)+'_new')
# for key in name_map.keys():
#     shutil.copy(os.path.join(mask_dir, key) + '.png', os.path.join(mask_dir+'_new', name_map[key]) + '.png')
# os.makedirs(os.path.join(normal_dir)+'_new')
# for key in name_map.keys():
#     shutil.copy(os.path.join(normal_dir, key) + '.png', os.path.join(normal_dir+'_new', name_map[key]) + '.png')
# os.makedirs(os.path.join(render_dir)+'_new')
# os.makedirs(os.path.join(albedo_dir)+'_new')

################### for train
# for shoe_type_index, shoe_type in enumerate(self.shoe_types):
#     for index, shoe_file in enumerate(self.render_paths[shoe_type_index]):
#         dir, file = shoe_file.rsplit('/', 1)
#         shutil.copy(shoe_file, os.path.join(dir + '_new', name_map[file[:110]] + '_light_' + file[133:135] + '.png'))
# for shoe_type_index, shoe_type in enumerate(self.shoe_types):
#     for index, (albedo_file, render_file) in enumerate(zip(self.albedo_paths[shoe_type_index], self.render_paths[shoe_type_index])):
#         dir, file = render_file.rsplit('/', 1)
#         shutil.copy(albedo_file, os.path.join(albedo_dir + '_new', name_map[file[:110]] + '_light_' + file[133:135] + '.png'))

############### for val_patches
# for shoe_type_index, shoe_type in enumerate(self.shoe_types):
#     for index, shoe_file in enumerate(self.render_paths[shoe_type_index]):
#         dir, file = shoe_file.rsplit('/', 1)
#         shutil.copy(shoe_file, os.path.join(dir + '_new', name_map[file[:112]] + '_light_' + file[135:137] + '.png'))
#
# for shoe_type_index, shoe_type in enumerate(self.shoe_types):
#     for index, (albedo_file, render_file) in enumerate(
#             zip(self.albedo_paths[shoe_type_index], self.render_paths[shoe_type_index])):
#         dir, file = render_file.rsplit('/', 1)
#         shutil.copy(albedo_file,
#                     os.path.join(albedo_dir + '_new', name_map[file[:112]] + '_light_' + file[135:137] + '.png'))
