import torch
from torch.autograd import Variable
import numpy as np, os
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
from torchvision.utils import save_image



eps = 0.0000000001



jet_cmap = matplotlib.cm.get_cmap('jet')
jet_cmap = jet_cmap(np.arange(256))[:, :3]
jet_cmap = jet_cmap[:, [2, 1, 0]]
jet_cmap2 = jet_cmap.copy()
jet_cmap2[:, 0] = np.arange(256)/255.0

viridis_cmap = matplotlib.cm.get_cmap('viridis')
viridis_cmap = viridis_cmap(np.arange(256))[:, :3]
viridis_cmap = viridis_cmap[:, [2, 1, 0]]

'''Get depth image in viridis color map'''
def get_color_mapped_images(depth, mask=None, cmap='viridis', mask_color=1, original_scale=False, to_tensor=None):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().detach().numpy()
        if mask is not None:
            mask = mask.squeeze().cpu().detach().numpy()
        to_tensor = True
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
    if mask is not None:
        if not original_scale:
            min_ = np.min(depth[mask])
            # min_ = 0.6
            max_ = np.max(depth[mask])
            depth = (depth - min_)/(max_ - min_)
        depth[~mask] = 1
    if cmap is None or cmap is 'viridis':
        cmap = viridis_cmap
    elif cmap is 'jet':
        cmap = jet_cmap
    elif cmap is 'jet2':
        cmap = jet_cmap2
    depth = (depth *255).astype(np.uint8)
    result = cmap[depth, :]
    if mask is not None:
        result[~mask[:,:, np.newaxis].repeat(3, axis= 2)] = mask_color
    if to_tensor:
        result = torch.tensor(result.transpose(2, 0, 1)).unsqueeze(0)
    return result


"""Given a tensor, make a one hot version of it if len(shape) == 1. otherwise, return the tensor"""
def make_one_hot(light):

    l = torch.zeros((light.shape[0], 17)).to(light.device)
    l[[i for i in range(light.shape[0])], light] = torch.ones(light.shape[0]).to(light.device)
    return l


def get_normal_visual(normal):
    return normal / 2 + 0.5

def make_variable(tensor, volatile=False, requires_grad=True):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    if volatile:
        requires_grad = False
    return Variable(tensor, volatile=volatile, requires_grad=requires_grad)

"""Return a placement tensor for an invalid tensor"""
def get_invalid_tensor(tensor=True):
    if tensor:
        return torch.Tensor([-1]).int()
    else:
        return -1


"""Check if a tensor is valid"""
def valid_tensor(tensor):
    return not invalid_tensor(tensor)


"""Check if a tensor is invalid"""
def invalid_tensor(tensor):
    return tensor is None or len(tensor.shape) == 1 and tensor.item() == -1

"""Create a gaussian kernel"""
def gaussian_kernel(kernlen=128, nsig=1):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

'''Create folders by the name of data (dictionary) keys and save images in data values in the corresponding folders.'''
def save_individual_images(data, folder, name, pad_h_before=None, pad_h_after=None, pad_w_before=None, pad_w_after=None, BGR_to_RGB=True):

    for title in data.keys():
        image = data[title]
        title = title.split(",")[0]
        if image.shape[1] == 3:
            if BGR_to_RGB:
                ind = [2, 1, 0]
            else:
                ind = [0, 1, 2]
        else:
            ind = [0]
        for i in range(image.shape[0]):
            if pad_h_before == None or (pad_h_before == 0 and pad_w_before == 0):
                sub_image = image[i, ind, ...]
            else:
                sub_image = image[i, ind, pad_h_before[i]:-pad_h_after[i], pad_w_before[i]:-pad_w_after[i]]
            os.makedirs(os.path.join(folder, title), exist_ok=True)

            n = name[i] + ".png" if len(name[i]) < 5 or name[i][-4:] not in [".png", ".jpg"] else name[i]
            save_image(sub_image.float(), os.path.join(folder, title, n))



def save_tensor_grid(data, save_path, BGR_to_RGB=True, fig_shape='square', figsize=None):
    show_tensor_grid(data, fig_shape=fig_shape, figsize=figsize)
    plt.savefig(save_path)
    plt.close()

def show_tensor_grid(data, BGR_to_RGB=True, fig_shape='square', figsize=None):
    if fig_shape =='square':
        fig_shape = (int(np.ceil(np.sqrt(len(data)))), int(np.ceil(np.sqrt(len(data)))))
    elif fig_shape == 'line':
        fig_shape = (len(data), 1)
    else:
        assert fig_shape[0]*fig_shape[1] >= len(data)
    if figsize is None:
        figsize = (10, 10)

    fig, axs = plt.subplots(fig_shape[0], fig_shape[1], figsize=figsize)
    for index, title in enumerate(data.keys()):
        ax_coord = np.unravel_index(index, fig_shape)
        image = get_image_from_tensor(data[title])
        if BGR_to_RGB and len(image.shape) == 3:
            image = image[..., [2, 1, 0]]
        if 1 in fig_shape:
            axs[ax_coord[0]].imshow(image)
            axs[ax_coord[0]].set_title(title)
        else:
            axs[ax_coord[0], ax_coord[1]].imshow(image)
            axs[ax_coord[0], ax_coord[1]].set_title(title)
    fig.tight_layout()
    return fig

def get_image_from_tensor(tensor, index=None, gray=False):
    if index is not None:
        t = tensor[index, ...].cpu().detach() # *255
    else:
        h = tensor.shape[2]
        w = tensor.shape[3]
        count = tensor.shape[0]
        t = torch.zeros((tensor.shape[1], h, count*w))

        for i in range(tensor.shape[0]):
            t[:,:, i*w:(i+1)*w] = tensor[i, ...]
        t = t.cpu().detach()
    if t.shape[0] == 1:
        if gray:
            t = t.repeat(3, 1, 1)
            t = t.transpose(0, 2).transpose(1, 0)
        else:
            t = t[0, ...]
    else:
        t = t.transpose(0, 2).transpose(1, 0)
    t = t.numpy()
    return t

def show_tensor(tensor, index=None, gray=False, figsize=None):
    t = get_image_from_tensor(tensor, index=index, gray=gray)
    plt.figure(figsize=figsize)
    plt.imshow(t)