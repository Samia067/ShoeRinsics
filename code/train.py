import os
import os.path
import random
from collections import deque
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import sklearn
from collections import  OrderedDict
from torch.utils.data import DataLoader

from code.util.option import Options
from code.model.models import get_model
from code.model.discriminator import Discriminator
from code.dataset.syn_real_dataset import SynRealDataset
from code.dataset.syn_dataset import SynDataset
from code.dataset.real_dataset import RealDataset
from code.util.morphology import Dilation2d
from code.util.misc import make_variable, get_normal_visual, make_one_hot, show_tensor, save_tensor_grid
from code.util.evaluation import iou, get_print

# from code.cycada.cycada.models.models import models
# from code.utils.utils import valid_tensor, save_individual_images
# from code.cycada.cycada.models.unet import GradientRegularizationLoss
# from code.utils.utils import plot_confusion_matrix, make_one_hot


def discriminator_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss(weight=weights, reduction='mean', ignore_index=255)
    loss = loss_fn_(F.log_softmax(score, dim=1), label)
    return loss

def get_sup_losses(output_syn, syn_labels, syn_mask, image_loss, class_loss):
    loss = [None] * len(output_syn)
    for output_index, output in enumerate(output_syn):
        if output is not None:
            # l1 loss if it is an image
            if len(output.shape) == 4:
                # use mask only if it is not light prediction - light prediction has ratio 1:2
                if output.shape[2] == output.shape[3]:
                    if syn_mask.shape[1] != output.shape[1]:
                        m = syn_mask.repeat((1, output.shape[1], 1, 1))
                    else:
                        m = syn_mask
                    if (syn_labels[output_index] == -1).any():
                        invalid_index = (syn_labels[output_index] == -1)
                        syn_labels[output_index][invalid_index] = output[invalid_index]
                    loss[output_index] = image_loss(output[m], syn_labels[output_index][m])

                else:
                    loss[output_index] = image_loss(output, syn_labels[output_index])
            else:
                loss[output_index] = class_loss(output, syn_labels[output_index])
    return loss

def decompose_and_recompose(decomposer, renderer, image, mask, light_sample='gumble'):

    output, feat = decomposer(image)

    feat1 = torch.cat(tuple((f for f in feat[:-1] if f is not None)))
    feat2 = feat[-1].unsqueeze(2).unsqueeze(3)
    feat = [feat1, feat2]

    if renderer is not None:
        renderer_inputs = process_renderer_input([output[0], output[1], output[2], output[4]], mask, light_sample=light_sample)
        if light_sample == 'max':
            renderer_inputs[3] = make_one_hot(renderer_inputs[3])
        reconstruction = renderer(renderer_inputs[0], renderer_inputs[1], renderer_inputs[2], renderer_inputs[3])
    else:
        reconstruction = None

    return output, feat, reconstruction

def get_visuals(net, image, mask, labels, label_names, label_prefix, light_visuals, visuals=None, subtract_min_depth=False, visualize_normals=True, renderer=None, light_sample='max'):
    if visuals is None:
        visuals = OrderedDict()

    outputs, features = net(image)
    features = torch.cat(tuple((f for f in features if f is not None)))


    if renderer is not None:
        renderer_inputs = process_renderer_input([outputs[0], outputs[1], outputs[2], outputs[4]], mask, light_sample=light_sample)
        if light_sample == 'max':
            renderer_inputs[3] = make_one_hot(renderer_inputs[3])
        reconstruction = renderer(renderer_inputs[0], renderer_inputs[1], renderer_inputs[2], renderer_inputs[3])
        reconstruction[~mask.repeat(1, reconstruction.shape[1], 1, 1)] = 1
        visuals[label_prefix + ' reconstruction'] = reconstruction

    for output_index, pred in enumerate(outputs):
        # if it is an image
        if pred is not None and label_names[output_index] in ['albedo', 'depth', 'normal']: # len(pred.shape) == 4:
            # pred = outputs[output_index]
            if label_names[output_index] == 'normal' and visualize_normals:
                pred = get_normal_visual(pred)
            if label_names[output_index] == 'depth' and subtract_min_depth:
                pred = pred - torch.min(pred)
            # if pred.shape[2] == pred.shape[3] or pred.shape[2] > 256:
            pred[~mask.repeat(1, pred.shape[1], 1, 1)] = 1
            visuals[label_prefix + ' ' + label_names[output_index] + ' pred'] = pred
            if labels is not None:
                if label_names[output_index] == 'normal':
                    label = get_normal_visual(labels[output_index])
                else:
                    label = labels[output_index]
                visuals[label_prefix + ' ' + label_names[output_index] + ' target'] = label
        elif pred is not None and label_names[output_index] == 'light_type':
            pred_light_type = process_renderer_input(list(outputs), mask, light_sample=light_sample)[-1]
            light_pred = light_visuals[pred_light_type]
            if len(light_pred.shape) == 6:
                light_pred_whole = torch.ones(light_pred.shape[0], 3, light_pred.shape[1]*light_pred.shape[4], light_pred.shape[2]*light_pred.shape[5]).to(light_pred.device)
                for i in range(light_pred.shape[1]):
                    for j in range(light_pred.shape[2]):
                        light_pred_whole[:, :, i*light_pred.shape[4]:(i+1)*light_pred.shape[4], j*light_pred.shape[5]:(j+1)*light_pred.shape[5]] = light_pred[:,i, j, ...]
                light_pred = light_pred_whole
            visuals[label_prefix + ' ' + label_names[output_index] + ' pred'] = light_pred
            if labels is not None:
                visuals[label_prefix + ' ' + label_names[output_index] + ' target'] = light_visuals[labels[output_index]]


    return visuals, features


def prepare_datasets(opt):
    # dataset for training
    opt.phase = 'train'
    train_dataset = SynRealDataset(opt)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch, shuffle=True,
                                                   num_workers=opt.num_workers)

    # syn val dataset
    opt.phase = 'val_patches'
    syn_val_dataset = SynDataset(os.path.join(opt.dataroot, opt.syn_train_dataset_dir, opt.phase), full=True)
    syn_val_dataloader = DataLoader(syn_val_dataset, batch_size=opt.batch, shuffle=False,
                                                    num_workers=opt.num_workers)

    # real val dataset
    real_val_dataset = RealDataset(os.path.join(opt.dataroot, opt.real_train_dataset_dir, opt.phase), full=True)
    real_val_dataloader = DataLoader(real_val_dataset, batch_size=opt.batch, shuffle=False,
                                                    num_workers=opt.num_workers)

    return train_dataloader, syn_val_dataloader, real_val_dataloader

def process_renderer_input(preds, mask, light_sample='gumble'):

    for i in range(3):
        if mask.shape[1] != preds[i].shape[1]:
            m = mask.repeat((1, preds[i].shape[1], 1, 1))
        else:
            m = mask
        preds[i][~m] = 1


    if light_sample == 'gumble':
        preds[-1] = nn.functional.gumbel_softmax(preds[-1], tau=0.1, hard=True)
    else:
        preds[-1] = torch.argmax(preds[-1], dim=1)
    return preds

"""Filter out synthetic image translations that diverge too far from the original synthetic images."""
def good_translations(syn_val_image_original, syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_light, syn_val_light_type, syn_val_mask, syn_val_name):
    diff = abs(syn_val_image - syn_val_image_original)
    diff = torch.max(diff.view(diff.shape[0], diff.shape[1], -1), dim=1)[0].view(diff.shape[0], 1, diff.shape[2], diff.shape[3])
    indices_to_keep = []
    for i in range(syn_val_image.shape[0]):
        if not ((torch.sum(diff[i] > translations_filter_max_threshold) > translations_filter_max_threshold_pixels) \
                or torch.mean(diff[i][syn_val_mask[i].expand_as(diff[i])]) > translations_filter_mean_threshold):
            indices_to_keep.append(i)
    syn_val_name = list(syn_val_name[i] for i in indices_to_keep)
    syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_light, syn_val_light_type, syn_val_mask = \
        [item[indices_to_keep, ...] for item in
         [syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_light, syn_val_light_type, syn_val_mask]]
    return syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_light, syn_val_light_type, syn_val_mask, syn_val_name

def shrink_mask(syn_val_image_original, syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_mask, dilate):
    syn_val_mask = ~((torch.sum(syn_val_image, dim=1) > 2.9999).unsqueeze(1) + ~syn_val_mask)
    # shrink mask
    syn_val_mask = ~(dilate((~syn_val_mask).float()).bool())
    for item in [syn_val_image_original, syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal]:
        item[~syn_val_mask.expand_as(item)] = 1
    return syn_val_image_original, syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_mask


translations_filter_max_threshold = 0.3
translations_filter_mean_threshold = 0.2
translations_filter_max_threshold_pixels = 200


def translate(generator, syn_image, syn_albedo, syn_depth, syn_normal, syn_light,
                          syn_light_type, syn_mask, syn_name, dilate):
    if generator is None:
        return syn_image, syn_albedo, syn_depth, syn_normal, syn_light, syn_light_type, syn_mask, syn_name
    # get translated synthetic image
    syn_image_original = syn_image
    syn_image = generator(syn_image)
    # shrink mask and overwrite outside mask for synthetic images to remove artifacts
    # near mask edge and outside the mask which are created when translating images
    syn_image[~syn_mask.repeat((1, 3, 1, 1))] = 1
    syn_image_original, syn_image, syn_albedo, syn_depth, syn_normal, syn_mask = \
        shrink_mask(syn_image_original, syn_image, syn_albedo, syn_depth, syn_normal, syn_mask, dilate)
    # filter out translations that diverge too far from the original synthetic images
    syn_image, syn_albedo, syn_depth, syn_normal, syn_light, syn_light_type, syn_mask, syn_name = \
        good_translations(syn_image_original, syn_image, syn_albedo, syn_depth, syn_normal, syn_light,
                          syn_light_type, syn_mask, syn_name)
    return syn_image, syn_albedo, syn_depth, syn_normal, syn_light, syn_light_type, syn_mask, syn_name

def syn_validation(opt, losses, syn_val_dataloader, net, renderer, discriminator, generator, dilate, l1_loss, CE_loss,
                   syn_label_names, dis_syn_val_label, device):
    syn_val_losses_list = [None] * 5
    for i in range(5):
        syn_val_losses_list[i] = deque()
    syn_val_losses_dis = deque()
    syn_val_acc_dis = deque()
    light_type_labels = deque()
    light_type_preds = deque()
    print_ious = deque()
    for syn_val_data in syn_val_dataloader:
        syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_light, syn_val_light_type, syn_val_mask, syn_val_name = syn_val_data
        syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_light, syn_val_light_type, syn_val_mask = [
            make_variable(item, requires_grad=False).to(device)
            for item in
            [syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_light, syn_val_light_type,
             syn_val_mask]]

        # translate synthetic images
        syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, \
        syn_val_light, syn_val_light_type, syn_val_mask, syn_val_name = \
            translate(generator, syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal,
                      syn_val_light, syn_val_light_type, syn_val_mask, syn_val_name, dilate)

        if syn_val_image.shape[0] == 0:
            continue

        syn_val_targets = [syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_light, syn_val_light_type]

        # process syn_val images for syn val data
        syn_val_preds, syn_val_features, _ = decompose_and_recompose(net, renderer, syn_val_image, syn_val_mask,
                                                                     light_sample='max')
        syn_val_print = get_print(syn_val_depth, syn_val_mask, None)
        syn_val_print_pred = syn_val_preds[1] > 0.2
        syn_val_print_pred[~syn_val_mask] = True
        for i in range(syn_val_print_pred.shape[0]):
            print_iou = iou(syn_val_print_pred[i:i + 1, ...], syn_val_print[i:i + 1, ...],
                            syn_val_mask[i:i + 1, ...]).item()
            print_ious.append(print_iou)
        syn_val_losses = get_sup_losses(syn_val_preds, syn_val_targets, syn_val_mask, l1_loss, CE_loss)
        for loss_index, l in enumerate(syn_val_losses):
            if l is not None:
                syn_val_losses_list[loss_index].append(l.item())

        # light
        light_type_labels.extend(syn_val_light_type.cpu().detach().numpy())
        light_type_preds.extend(torch.argmax(syn_val_preds[4], axis=1).cpu().detach().numpy())

        # get discriminator loss for syn val translated images
        dis_score_syn_val = discriminator(syn_val_features[0],
                                          syn_val_features[1] if len(syn_val_features) > 1 else None)
        if dis_syn_val_label is None or dis_syn_val_label.shape[0] != dis_score_syn_val.shape[0]:
            batch_syn, _, h, w = dis_score_syn_val.size()
            dis_syn_val_label = make_variable(torch.ones(batch_syn, h, w).long(), requires_grad=False).to(device)

        # compute loss for discriminator
        loss_dis = discriminator_loss(dis_score_syn_val, dis_syn_val_label)
        syn_val_losses_dis.append(loss_dis.item())

        # compute accuracy for discriminator
        pred_dis = torch.squeeze(dis_score_syn_val.max(1)[1])
        acc_dis = (pred_dis == dis_syn_val_label).float().mean().item()
        syn_val_acc_dis.append(acc_dis)
    if opt.train_discriminator:
        losses['discriminator_syn_val_losses'] = np.mean(syn_val_losses_dis)
        losses['discriminator_syn_val_accuracy'] = np.mean(syn_val_acc_dis)
    loss_total = None
    for loss_index, l in enumerate(syn_val_losses):
        if l is not None:
            l_mean = np.mean(syn_val_losses_list[loss_index]) * opt.syn_weights[loss_index]
            losses['syn val loss/' + str(syn_label_names[loss_index])] = l_mean
            loss_total = loss_total + l_mean if loss_total is not None else l_mean

    losses['syn val loss/total'] = np.mean(loss_total.item())
    losses['syn val loss/print_iou'] = np.mean(print_ious)
    return loss_dis, dis_syn_val_label

def real_validation(opt, losses, reconstruct, real_val_dataloader, net, renderer, discriminator,
                    l1_loss, unsup_loss_names, dis_real_val_label, device):
    if opt.train_discriminator or reconstruct:
        real_val_losses_list = [None] * 2
        for i in range(2):
            real_val_losses_list[i] = deque()
        real_val_losses_dis = deque()
        real_val_acc_dis = deque()
        for real_val_data in real_val_dataloader:
            real_val_image, real_val_albedo_segmentation, real_val_mask, real_val_name = real_val_data
            real_val_image, real_val_mask, real_val_albedo_segmentation = [
                make_variable(item, requires_grad=False).to(device) for item in
                [real_val_image, real_val_mask, real_val_albedo_segmentation]]

            # get discriminator loss and unsupervised losses for real val images
            real_val_preds, real_val_features, real_val_reconstruction = decompose_and_recompose(net, renderer,
                                                                                                 real_val_image,
                                                                                                 real_val_mask,
                                                                                                 light_sample='max')

            if real_val_reconstruction is not None:
                preds = [real_val_preds[0], real_val_reconstruction]
                pseudo_labels = [real_val_albedo_segmentation, real_val_image]
                real_val_losses = get_sup_losses(preds, pseudo_labels, real_val_mask, l1_loss, None)
                for loss_index, l in enumerate(real_val_losses):
                    if l is not None:
                        real_val_losses_list[loss_index].append(l.item())

            dis_score_real_val = discriminator(real_val_features[0],
                                               real_val_features[1] if len(real_val_features) > 1 else None)
            if dis_real_val_label is None or dis_real_val_label.shape[0] != dis_score_real_val.shape[0]:
                batch_syn, _, h, w = dis_score_real_val.size()
                dis_real_val_label = make_variable(torch.zeros(batch_syn, h, w).long(), requires_grad=False).to(device)

            # compute loss for discriminator
            loss_dis = discriminator_loss(dis_score_real_val, dis_real_val_label)
            real_val_losses_dis.append(loss_dis.item())

            # compute accuracy for discriminator
            pred_dis = torch.squeeze(dis_score_real_val.max(1)[1])
            acc_dis = (pred_dis == dis_real_val_label).float().mean().item()
            real_val_acc_dis.append(acc_dis)

        if opt.train_discriminator:
            losses['discriminator_real_val_losses'] = np.mean(real_val_losses_dis)
            losses['discriminator_real_val_accuracy'] = np.mean(real_val_acc_dis)

        if reconstruct:
            loss_total = None
            for loss_index, l in enumerate(real_val_losses):
                if l is not None:
                    l_mean = np.mean(real_val_losses_list[loss_index]) * opt.real_weights[loss_index]
                    losses['real val loss/' + str(unsup_loss_names[loss_index])] = l_mean
                    loss_total = loss_total + l_mean if loss_total is not None else l_mean
            losses['real val loss/total'] = np.mean(loss_total.item())


    if opt.train_discriminator:
        losses['discriminator_average_val_losses'] = (losses['discriminator_real_val_losses'] + losses[
            'discriminator_syn_val_losses']) / 2
        losses['discriminator_average_val_accuracy'] = (losses['discriminator_real_val_accuracy'] + losses[
            'discriminator_syn_val_accuracy']) / 2
        losses['discriminator_average_val_accuracy - 0.5'] = abs(0.5 - losses['discriminator_average_val_accuracy'])
    return loss_dis, dis_real_val_label

def save_models(opt, net, opt_rep, discriminator, opt_dis, loss_dis, iteration, best):
    if opt.train_net:
        net.save_checkpoint(opt.output, opt_rep, iteration, None, None, is_best=best)
    if opt.train_discriminator:
        label = 'best' if best else 'itercurr'
        torch.save({
            'iteration': iteration,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': opt_dis.state_dict(),
            'loss': loss_dis,
        }, opt.output + '/discriminator-'+label+'.pth')

def validate(opt, iteration, losses, syn_val_dataloader, real_val_dataloader, net, opt_rep, renderer, opt_ren,
             discriminator, opt_dis, generator, dilate, l1_loss, CE_loss, syn_label_names, unsup_loss_names,
             reconstruct, dis_syn_val_label, dis_real_val_label,
             best_renderer_error, best_syn_print_iou, best_discrim_acc_diff, device):
    loss_dis = None
    with torch.no_grad():
        if opt.train_renderer:
            renderer.eval()
            syn_val_losses_list = deque()

            for syn_val_data in syn_val_dataloader:
                syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_light, syn_val_light_type, syn_val_mask, syn_val_name = syn_val_data
                syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_light, syn_val_light_type, syn_val_mask = [
                    make_variable(item, requires_grad=False).to(device)
                    for item in
                    [syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, syn_val_light, syn_val_light_type,
                     syn_val_mask]]

                # translate synthetic images
                syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal, \
                syn_val_light, syn_val_light_type, syn_val_mask, syn_val_name = \
                    translate(generator, syn_val_image, syn_val_albedo, syn_val_depth, syn_val_normal,
                              syn_val_light, syn_val_light_type, syn_val_mask, syn_val_name, dilate)
                if syn_val_image.shape[0] == 0:
                    continue

                reconstruction = renderer(syn_val_albedo, syn_val_depth, syn_val_normal, make_one_hot(syn_val_light_type))
                syn_val_losses_list.append(
                    get_sup_losses([reconstruction], [syn_val_image], syn_val_mask, l1_loss, None)[0].item())

            losses['syn val loss/render'] = np.mean(syn_val_losses_list)

            # save best renderer
            if losses['syn val loss/render'] < best_renderer_error:
                best_renderer_error = losses['syn val loss/render']
                renderer.save_checkpoint(opt.output, opt_ren, iteration, None, None, is_best=True)
        else:
            net.eval()
            # synthetic validation
            loss_dis_syn, dis_syn_val_label = syn_validation(opt, losses, syn_val_dataloader, net, renderer, discriminator,
                           generator, dilate, l1_loss, CE_loss, syn_label_names, dis_syn_val_label, device)

            if opt.train_discriminator:
                # real validation
                loss_dis_real, dis_real_val_label = real_validation(opt, losses, reconstruct, real_val_dataloader,
                                net, renderer, discriminator, l1_loss, unsup_loss_names, dis_real_val_label, device)

                loss_dis = (loss_dis_real + loss_dis_syn)/2
                losses['average val loss/total'] = (losses['syn val loss/total'] + losses['real val loss/total']) / 2
                criteria = abs(0.5 - losses['discriminator_average_val_accuracy'])
                if criteria <= best_discrim_acc_diff:
                    print("Saving best model at ", iteration)
                    best_discrim_acc_diff = criteria
                    save_models(opt, net, opt_rep, discriminator, opt_dis, loss_dis, iteration, True)

            else:
                # save best model
                if best_syn_print_iou <= losses['syn val loss/print_iou']:
                    best_syn_print_iou = losses['syn val loss/print_iou']
                    save_models(opt, net, opt_rep, discriminator, opt_dis, loss_dis, iteration, True)

    return best_renderer_error, best_syn_print_iou, best_discrim_acc_diff, loss_dis, dis_syn_val_label, dis_real_val_label

# @click.command()
# @click.argument('output')
# @click.option('--dataset', required=True, multiple=True)
# @click.option('--datadir', default="", type=click.Path(exists=True))
# @click.option('--momentum', '-m', default=0.9)
# @click.option('--snapshot', '-s', default=5000)
# @click.option('--downscale', type=int)
# @click.option('--crop_size', default=None, type=int)
# @click.option('--half_crop', default=None)
# @click.option('--cls_weights', type=click.Path(exists=True))
# @click.option('--weights_discrim', type=click.Path(exists=True))
# @click.option('--weights_init', type=click.Path(exists=True))
# @click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
# @click.option('--lsgan/--no_lsgan', default=False)
# @click.option('--num_cls', type=int, default=1)
# @click.option('--gpu', default='0')
# @click.option('--max_iter', default=10000)
# @click.option('--lambda_d', default=1.0)
# @click.option('--lambda_g', default=1.0)
#
# # extra parameters for syn_depth estimation
# @click.option('--real_with_gt_dataset_dir', default='real_test_shoes', help="real dataset with GT")
# @click.option('--full/--patch', default=False, help="whether or not to load full images instead of patches")
# @click.option('--dataset_mode', default='syn_real')
# @click.option('--dataroot', default='/home/sshafiqu/Research/Shoeprint/dataset/')
# @click.option('--phase', default='train', type=click.Choice(['train', 'val', 'test']))
# @click.option('--serial_batches', default=False)
# @click.option('--reset_opt_dis', default=False)
# @click.option('--reset_opt_rep', default=False)
# @click.option('--evaluation_set', default='sneakers', type=click.Choice(['sneakers', 'all', 'val', 'test']))
# @click.option('--conv', default=False)
# @click.option('--discrim_kernel_size', type=int, default=3)
# @click.option('--threshold', type=int, default=55)
# @click.option('--num_workers', type=int, default=0)
# @click.option('--weights_renderer', default=None, type=click.Path(exists=True))
# @click.option('--renderer_mode', default='3', type=click.Choice(['2', '3']))
# @click.option('--gen_eval', default=False)
# @click.option('--syn_original', default=False)
# @click.option('--filter_translations', default=False)
# @click.option('--weight_decay_albedo', type=float, default=1)
# @click.option('--weight_decay_normal', type=float, default=1)
# @click.option('--weight_decay_light', type=float, default=1)
# @click.option('--weight_decay_iteration', type=int, default=1000)
# @click.option('--lr_rep', type=float, default=1e-3)
# @click.option('--betas_rep', default='0.5,0.999')
# @click.option('--see_real_test_data', default=False)

def main():
    # parse command line arguments
    opt = Options(train=False).parse()

    # Set seed so that visuals is sampled in a consistent way
    np.random.seed(1337)
    random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load renderer if needed
    reconstruct = opt.weights_renderer is not None or opt.train_renderer
    if reconstruct:
        renderer = get_model('renderer', weights_init=opt.weights_renderer).to(device)
        renderer.eval()
    else:
        renderer = None

    opt_ren = torch.optim.Adam(renderer.parameters(), lr=1e-3, betas=(0.5, 0.999)) if opt.train_renderer else None

    if opt.weights_translator is not None:
        generator = get_model('gen', pretrained=True, weights_init=opt.weights_translator).to(device)

        for param in generator.parameters():
            param.requires_grad = False
    else:
        generator = None


    net = get_model('decomposer', weights_init=opt.weights_decomposer, output_last_ft=True, out_range='0,1').to(device)

    pretrained_discrim = not (opt.weights_discriminator==None)
    discriminator = Discriminator(input_dim=32, output_dim=2, pretrained=pretrained_discrim,
            weights_init=opt.weights_discriminator).to(device)

    # setup optimizers
    opt_dis = torch.optim.SGD(discriminator.parameters(), lr=opt.lr, momentum=0.99, weight_decay=0.0005)
    opt_rep = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.5, 0.999))

    iteration = 0
    num_update_g = 0
    last_update_g = -1
    losses_super = deque(maxlen=100)
    losses_unsupervised = deque(maxlen=100)
    loss_sup_plus_unsup = deque(maxlen=100)
    losses_dis = deque(maxlen=100)
    losses_rep = deque(maxlen=100)
    accuracies_dom = deque(maxlen=100)
    max_iter = 100000
    print('max iter:', max_iter)
    os.makedirs(opt.output, exist_ok=True)
   
    net.train()
    discriminator.train()
    l1_loss = torch.nn.L1Loss()
    CE_loss = torch.nn.CrossEntropyLoss()

    dom_acc_thresh = 55
    dis_label_concat = target_dom_fake_real = None
    acc_dom_mean = 100

    losses = OrderedDict()
    target_optimized = False
    dis_syn_val_label = dis_real_val_label = None

    train_dataloader, syn_val_dataloader, real_val_dataloader = prepare_datasets(opt)

    syn_label_names = ['albedo', 'depth', 'normal', 'light', 'light_type']
    unsup_loss_names = ['albedo segments', 'reconstruction']


    best_discrim_acc_diff = best_renderer_error = np.inf
    best_syn_print_iou = 0
    dilate = Dilation2d(1, 1, kernel_size=7, soft_max=False).to(device)

    lambda_g = 0.1
    lambda_d = 1
    while iteration < max_iter:

        for data in train_dataloader:
            syn_image, syn_albedo, syn_depth, syn_normal, syn_light, syn_light_type, syn_mask, syn_name, real_image, real_albedo, real_mask, real_name = data


            ###########################
            # 1. Setup Data Variables #

            syn_image, syn_depth, syn_albedo, syn_normal, syn_light, syn_light_type, syn_mask, \
            real_image, real_albedo, real_mask = [make_variable(item, requires_grad=False).to(device) for item in
                                                   [syn_image, syn_depth, syn_albedo, syn_normal, syn_light,
                                                    syn_light_type, syn_mask, real_image, real_albedo, real_mask]]

            # translate synthetic images
            syn_image, syn_albedo, syn_depth, syn_normal, syn_light, syn_light_type, syn_mask, syn_name = \
                translate(generator, syn_image, syn_albedo, syn_depth, syn_normal, syn_light,
                          syn_light_type, syn_mask, syn_name, dilate)

            if syn_image.shape[0] == 0:
                continue

            syn_targets = [syn_albedo, syn_depth, syn_normal, syn_light, syn_light_type]

            losses.clear()

            ###########################
            # Log and compute metrics #
            ###########################

            if iteration == 0 or ((opt.train_renderer or target_optimized) and iteration % 500 == 0):
                target_optimized = False
                best_renderer_error, best_syn_print_iou, best_discrim_acc_diff, loss_dis, dis_syn_val_label, dis_real_val_label = \
                    validate(opt, iteration, losses, syn_val_dataloader, real_val_dataloader, net, opt_rep,
                         renderer, opt_ren, discriminator, opt_dis, generator, dilate, l1_loss, CE_loss,
                         syn_label_names, unsup_loss_names, reconstruct, dis_syn_val_label, dis_real_val_label,
                         best_renderer_error, best_syn_print_iou, best_discrim_acc_diff, device)


            ################
            # Save outputs #
            ################

            # every 500 iters save current model
            if iteration % 500 == 0:
                save_models(opt, net, opt_rep, discriminator, opt_dis, loss_dis, iteration, False)


            #############################
            # 2. Optimize Discriminator #
            #############################

            if not opt.train_renderer and opt.train_discriminator:
                # zero gradients for optimizer
                opt_dis.zero_grad()
                opt_rep.zero_grad()
                discriminator.train()

                # extract features
                _, f_syn, _ = decompose_and_recompose(net, renderer, syn_image, syn_mask)
                f_syn = [Variable(f.data, requires_grad=False) for f in f_syn]
                dis_score_syn = discriminator(f_syn[0], f_syn[1] if len(f_syn) > 1 else None)

                _, f_real, _ = decompose_and_recompose(net, renderer, real_image, real_mask)
                f_real = [Variable(f.data, requires_grad=False) for f in f_real]
                dis_score_real = discriminator(f_real[0], f_real[1] if len(f_real) > 1 else None)

                dis_pred_concat = torch.cat((dis_score_syn, dis_score_real))

                # prepare real and fake labels
                if dis_label_concat is None or dis_label_concat.shape[0] != (dis_score_real.shape[0] + dis_score_syn.shape[0]):
                    batch_real,_,h,w = dis_score_real.size()
                    batch_syn,_,_,_ = dis_score_syn.size()
                    dis_label_concat = make_variable(torch.cat([torch.ones(batch_syn,h,w).long(),
                                                                torch.zeros(batch_real,h,w).long()]), requires_grad=False).to(device)

                # compute loss for discriminator
                loss_dis = discriminator_loss(dis_pred_concat, dis_label_concat)
                (lambda_d * loss_dis).backward()
                losses_dis.append(loss_dis.item())

                # optimize discriminator
                opt_dis.step()

                # compute discriminator acc
                pred_dis = torch.squeeze(dis_pred_concat.max(1)[1])
                dom_acc = (pred_dis == dis_label_concat).float().mean().item()
                accuracies_dom.append(dom_acc * 100.)
                acc_dom_mean = np.mean(accuracies_dom)
                loss_dom_mean = np.mean(losses_dis)

                # add discriminator info to log
                losses['discriminator_loss'] = loss_dom_mean
                losses['discriminator_acc'] = acc_dom_mean


            ###########################
            # Optimize Target Network #
            ###########################

            if opt.train_discriminator and opt.train_net and not opt.train_renderer and \
                    (acc_dom_mean > dom_acc_thresh and len(accuracies_dom) in [0, accuracies_dom.maxlen]):

                target_optimized = True
                last_update_g = iteration
                num_update_g += 1
                if num_update_g % 1 == 0:
                    print('Updating decomposer with adversarial loss ({:d} times)'.format(num_update_g))

                # zero out optimizer gradients
                opt_dis.zero_grad()
                opt_rep.zero_grad()
                net.train()

                _, f_real, _ = decompose_and_recompose(net, renderer, real_image, real_mask)
                dis_score_real = discriminator(f_real[0], f_real[1] if len(f_real) > 1 else None)

                # create fake label
                if target_dom_fake_real is None or target_dom_fake_real.shape[0] != dis_score_real.shape[0]:
                    batch,_,h,w = dis_score_real.size()
                    target_dom_fake_real = make_variable(torch.ones(batch,h,w).long(), requires_grad=False).to(device)

                # compute loss for target net
                loss_gan_real = discriminator_loss(dis_score_real, target_dom_fake_real)
                (lambda_g * loss_gan_real).backward()
                losses_rep.append(loss_gan_real.item())
                losses['generator_loss'] = np.mean(losses_rep)

                # optimize target net
                opt_rep.step()

            if opt.train_net and not opt.train_renderer and \
                    (acc_dom_mean > dom_acc_thresh and len(accuracies_dom) in [0, accuracies_dom.maxlen]):
                target_optimized = True
                print('Updating decomposer using source supervised and unsupervised losses.')

                # zero out optimizer gradients
                opt_dis.zero_grad()
                opt_rep.zero_grad()
                net.train()

                output_syn, _, _ = decompose_and_recompose(net, None, syn_image, syn_mask)

                # supervised losses on decomposer output
                loss_supervised = get_sup_losses(output_syn, syn_targets, syn_mask, l1_loss, CE_loss)
                loss_sup_total = None
                for output_index in range(len(output_syn)):
                    if loss_supervised[output_index] is not None:
                        losses['supervised_source_loss/' + syn_label_names[output_index]] = loss_supervised[output_index]*opt.syn_weights[output_index]
                        curr_loss = opt.syn_weights[output_index]*loss_supervised[output_index]
                        loss_sup_total = loss_sup_total + curr_loss if loss_sup_total is not None else curr_loss
                losses_super.append(loss_sup_total.item())
                losses['supervised_source_loss_total'] = np.mean(losses_super)

                # unsupervised losses on renderer output
                if reconstruct:
                    output_real, _, reconstruction = decompose_and_recompose(net, renderer, real_image, real_mask)
                    preds = [output_real[0], reconstruction]
                    pseudo_labels = [real_albedo, real_image]
                    loss_unsup = get_sup_losses(preds, pseudo_labels, real_mask, l1_loss, None)
                    loss_unsup_total = None
                    for loss_index in range(len(unsup_loss_names)):
                        if loss_unsup[loss_index] is not None:
                            losses['unsupervised_loss/' + unsup_loss_names[loss_index]] = loss_unsup[loss_index]*opt.real_weights[loss_index]
                            curr_loss = opt.real_weights[loss_index] * loss_unsup[loss_index]
                            loss_unsup_total = loss_unsup_total + curr_loss if loss_unsup_total is not None else curr_loss

                    losses_unsupervised.append(loss_unsup_total.item())
                    losses['unsupervised_loss_total'] = np.mean(losses_unsupervised)
                else:
                    loss_unsup_total = torch.autograd.Variable(torch.zeros(1).to(device))


                total_loss = loss_sup_total + loss_unsup_total
                loss_sup_plus_unsup.append(total_loss.item())
                total_loss.backward()
                losses['supervised_plus_unsupervised_loss_total'] = np.mean(loss_sup_plus_unsup)
                # optimize target net
                opt_rep.step()

            if opt.train_renderer:
                opt_ren.zero_grad()
                renderer.train()
                reconstruction = renderer(syn_albedo, syn_depth, syn_normal, make_one_hot(syn_light_type))
                loss_supervised = get_sup_losses([reconstruction], [syn_image], syn_mask, l1_loss, None)[0]
                losses_super.append(loss_supervised.item())
                losses['supervised_source_loss/render'] = np.mean(losses_super)
                loss_supervised.backward()
                opt_ren.step()

            ###########################
            # decay synthetic training weights
            weight_decay_iteration = 1000
            weight_decay_albedo = weight_decay_normal = weight_decay_light = 0.9
            if iteration != 0 and iteration % weight_decay_iteration == 0:
                print("weight decay")
                opt.syn_weights[0] *= weight_decay_albedo
                opt.syn_weights[2] *= weight_decay_normal
                opt.syn_weights[4] *= weight_decay_light

            ###########################
            # return if feature discriminator training not progressing well
            if iteration - last_update_g >= 8000*len(train_dataloader):
                print('No suitable discriminator found -- returning.')
                net.save_checkpoint(opt.output, opt_rep, iteration, None, None)
                return

            iteration += 1



if __name__ == '__main__':
    main()
