import os
import os.path
from collections import deque
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader

from code.util.option import Options
from code.util.misc import make_variable, valid_tensor, get_color_mapped_images, save_individual_images, save_tensor_grid
from code.util.augmentation import reverse_modification, get_image_modifications
from code.util.evaluation import get_print, iou
from code.model.models import get_model
from code.dataset.RealShoeDataset import RealShoeDataset


def get_average_visuals(net, image, mask, visuals=None, subtract_min_depth=True, conv=True, test_time_aug=False):
    # initialize dictionary to store output visuals
    if visuals is None:
        visuals = OrderedDict()

    # get test time image augmentations if needed
    # otherwise use original image only
    if test_time_aug:
        conv = False
        images, transforms = get_image_modifications(image)
        masks, _ = get_image_modifications(mask)
    else:
        images = [image]
        masks = [mask]
        transforms = ["original"]

    # compute and save albedo and depth (or mean albedo and depth for test time augmentation)
    albedos = torch.empty((0, 3, image.shape[2], image.shape[3])).to(image.device)
    depths = torch.empty((0, 1, image.shape[2], image.shape[3])).to(image.device)

    with torch.no_grad():
        for curr_image, cur_mask, transform in zip(images, masks, transforms):
            # compute model outputs for each image
            # cur_outputs, cur_features = net(curr_image)
            cur_outputs, cur_features = net(curr_image, mask=None if conv else cur_mask)
            albedo, depth, normal, light_env, light_id = cur_outputs

            # reverse image modification
            # when test time augmentation is not used, reverse_modification does not alter the model outputs
            if valid_tensor(albedo):
                cur_albedo = reverse_modification(albedo, transform, label='albedo', original_shape=image.shape[2:])
            cur_depth = reverse_modification(depth, transform, label='depth', original_shape=image.shape[2:])
            cur_depth[~mask] = 1

            # store albedo and depth predictions
            if valid_tensor(albedo):
                albedos = torch.cat((albedos, cur_albedo))
            depths = torch.cat((depths, cur_depth))

    # calculate mean albedo prediction
    if valid_tensor(albedo):
        albedo_std, albedo_mean = torch.std_mean(albedos, dim=0, keepdim=True)
        visuals['albedo pred'] = albedo_mean

    # calculate mean depth prediction
    depth_std, depth_mean = torch.std_mean(depths, dim=0, keepdim=True)
    if subtract_min_depth:
        depth_mean = depth_mean - torch.min(depth_mean)
        depth_mean[~mask.repeat(1, depth_mean.shape[1], 1, 1)] = 1
    visuals['depth pred'] = depth_mean

    return visuals


def prepare_datasets(opt):
    val_dataset_dir = os.path.join(opt.dataroot, opt.val_dataset_dir)
    val_dataset = RealShoeDataset(val_dataset_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=opt.num_workers)

    return val_dataloader


def main():
    # parse command line arguments
    opt = Options(train=False).parse()

    # Set seed so that visuals is sampled in a consistent way
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load decomposer
    net = get_model('decomposer', weights_init=opt.weights_decomposer, output_last_ft=True, out_range='0,1').to(device)
    net.eval()

    # load validation dataset
    val_dataloader = prepare_datasets(opt)

    image_dir = opt.val_dataset_dir + ("_test_time_aug" if opt.test_time_aug else "")
    os.makedirs(os.path.join(opt.output, image_dir, "grid"), exist_ok=True)
    print_ious = deque()
    iou_file_name = os.path.join(opt.output, image_dir + "_iou.txt")
    f = open(iou_file_name, "a")

    for data in val_dataloader:
        image, mask, shoeprint, albedo_segmentation, name, pad_h_before, pad_h_after, pad_w_before, pad_w_after = data
        image, mask, shoeprint = [make_variable(item, requires_grad=False).to(device) for item in [image, mask, shoeprint]]

        visuals = OrderedDict()
        visuals[name[0]] = image
        if valid_tensor(shoeprint):
            visuals['GT print'] = shoeprint
        visuals['mask'] = mask

        # get model predictions
        visuals = get_average_visuals(net, image, mask, visuals=visuals, test_time_aug=opt.test_time_aug)

        # get print predictions
        real_gt_print_pred = get_print(visuals['depth pred'], mask, shoeprint)
        real_gt_print_pred[~mask] = True
        print_iou = iou(real_gt_print_pred, shoeprint, mask).item()
        print_ious.append(print_iou)
        visuals['print pred, iou: {:0.2f}'.format(print_iou)] = real_gt_print_pred
        visuals['depth pred'] = get_color_mapped_images(visuals['depth pred'], mask).to(image.device, torch.float32)

        # save grid of outputs
        save_path = os.path.join(opt.output, image_dir, "grid", name[0])
        save_tensor_grid(visuals, save_path, fig_shape=[2, 3], figsize=(10, 5))

        # save individual images
        del visuals[name[0]]
        visuals['real image'] = image
        del visuals['print pred, iou: {:0.2f}'.format(print_iou)]
        visuals['print pred'] = real_gt_print_pred
        save_individual_images(visuals, os.path.join(opt.output, image_dir), name,
                               pad_h_before=pad_h_before, pad_h_after=pad_h_after,
                               pad_w_before=pad_w_before, pad_w_after=pad_w_after)

        f.write(name[0].rsplit('.', 1)[0] + '\t{:0.2f}\n'.format(print_iou))

    print(np.mean(print_ious))
    f.write('mean iou: {:0.2f}\n'.format(np.mean(print_ious)))
    f.close()

    return


if __name__ == '__main__':
    main()
