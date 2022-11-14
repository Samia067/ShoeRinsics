# Code ported from CyCADA
# Github repo: https://github.com/jhoffman/cycada_release


import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Code ported from CyCADA
# Github repo: https://github.com/jhoffman/cycada_release
class Discriminator(nn.Module):
    def __init__(self, input_dim=4096, output_dim=2, pretrained=False, weights_init='', kernel_size=3, light=True):
        super().__init__()
        dim1 = 1024 if input_dim==4096 else 64 # 512
        dim2 = int(dim1/2)
        self.D = nn.Sequential(
            nn.Conv2d(input_dim, dim1, kernel_size),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim2, kernel_size),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim2, output_dim, kernel_size)
            )
        self.light = light
        if self.light:
            self.D_light = nn.Sequential(
                nn.Conv2d(17, dim1, 1),
                nn.Dropout2d(p=0.5),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim1, dim2, 1),
                nn.Dropout2d(p=0.5),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim2, output_dim, 1)
            )

        if pretrained and weights_init is not None:
            self.load_weights(weights_init)

    def forward(self, x, light=None):
        d_score = self.D(x)
        if self.light:
            if len(light.shape) == 6:
                light = light.view(1, 17, light.shape[4], light.shape[5])
                d_light = self.D_light(light)
                d_light = d_light.repeat((1, 1, d_score.shape[2]//d_light.shape[2], d_score.shape[3]//d_light.shape[3]))
                d_light = nn.functional.pad(d_light, (d_score.shape[3] - d_light.shape[3],0, d_score.shape[2] - d_light.shape[2], 0))

            else:
                d_light = self.D_light(light)
                d_light = d_light.repeat((1, 1, d_score.shape[2], d_score.shape[3]))
            d_score = torch.cat((d_score, d_light))
        return d_score

    def load_weights(self, weights):
        print('Loading discriminator weights')
        data = torch.load(weights, map_location=device)
        self.load_state_dict(data['model_state_dict'])
        return data['iteration'], data['optimizer_state_dict'], data['loss']


