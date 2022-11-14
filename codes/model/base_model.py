import torch.nn as nn
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseModel(nn.Module):
    def __init__(self, name):
        super(BaseModel, self).__init__()
        self.name = name

    """
            Count total number of parameters for this model.
            """

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_count(self, print_count=False):
        count = 0
        million = 1000000
        for i, model in enumerate(self.parts):
            # p_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            p_num = model.param_count()
            if print_count:
                print(self.part_names[i], "has parameter count", p_num / million, "M")
            count += p_num
        return count

    def save_checkpoint(self, save_dir, optimizer, epoch, loss, error, is_best=False):
        if is_best:
            prefix = self.name + "_best_"
        else:
            prefix = self.name + "_"

        save_data = {}
        save_data['epoch'] = epoch
        save_data['loss'] = loss
        save_data['error'] = error
        save_data['state'] = self.state_dict()
        if optimizer is not None:
            save_data['optimizer'] = optimizer.state_dict()
        filename = os.path.join(save_dir, prefix + 'state.t7')
        torch.save(save_data, open(filename, 'wb'))

    def load_pretrained_model(self, load_dir, best=True):
        if os.path.isfile(load_dir):
            path = load_dir
        else:
            if best:
                prefix = self.name + "_best_"
            else:
                prefix = self.name + "_"
            path = os.path.join(load_dir, prefix + 'state.t7')
        print("loading: ", path)
        if torch.cuda.is_available():
            data = torch.load(path, map_location=device)
        else:
            data = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(data['state'])
        if 'optimizer' not in data:
            data['optimizer'] = None
        return data['epoch'], data['loss'], data['error'], data['optimizer']
