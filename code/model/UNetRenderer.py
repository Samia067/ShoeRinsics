import torch as th
from code.model.UNetDecomposer import *
from code.model.base_model import BaseModel
# from code.utils.utils import make_one_hot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LightEncoder(nn.Module):
    def __init__(self, h=64, w=128, n=4, conv_times=2, light_type_count=17):
        super(LightEncoder, self).__init__()
        self.h = h
        self.w = w
        self.kernel_size = 1
        self.c = n * 128
        self.inter_ch = n * 64

        self.light_type_count = light_type_count
        if conv_times == 1:
            self.classification = Conv(self.light_type_count, self.inter_ch // 2, kernel_size=self.kernel_size).to(device)
            self.in_conv = Conv(self.h * self.w, self.inter_ch // 2, kernel_size=self.kernel_size).to(device)
        else:
            self.classification = DoubleConv(self.light_type_count, self.inter_ch // 2, inter_ch=self.light_type_count, kernel_size=self.kernel_size).to(device)
            self.in_conv = DoubleConv(self.h * self.w, self.inter_ch // 2, inter_ch=self.light_type_count, kernel_size=self.kernel_size).to(device)

        self.conv = nn.Conv2d(self.inter_ch, self.inter_ch, kernel_size=self.kernel_size).to(device)

        # incoming_skip_channels = 1
        self.residual_blocks = nn.Sequential()
        self.pools = nn.Sequential()
        for i in range(7):
            self.residual_blocks.add_module(str(i), ResConv(self.inter_ch, self.inter_ch, kernel_size=self.kernel_size, conv_times=conv_times).to(device))
            self.pools.add_module(str(i), nn.MaxPool3d((self.inter_ch, 1, 1)))
            # self.pools.add_module(str(i), nn.MaxPool2d((self.inter_ch, self.inter_ch)))
        self.bn_lrelu = nn.Sequential(
            nn.BatchNorm2d(self.inter_ch),
            nn.LeakyReLU(0.1)).to(device)

        self.out_conv = Inconv(self.inter_ch, self.c // 2, kernel_size=self.kernel_size, conv_times=conv_times).to(device)



    def forward(self, light, light_type):
        light = th.reshape(light, (light.shape[0], -1, 1, 1))
        light = self.in_conv(light)

        light_type = th.reshape(light_type, (light_type.shape[0], light_type.shape[1], 1, 1))
        light_type = self.classification(light_type)
        x = torch.cat((light, light_type), dim=1)

        x = self.conv(x)

        skip_connections = []
        len_residual_blocks = len(self.residual_blocks)
        for i in range(len_residual_blocks):
            # x = self.residual_blocks[i](x)
            res_block_output = self.residual_blocks[i](x)
            x = x + res_block_output
            res_block_output = self.pools[i](res_block_output)
            spatial_repeat = np.power(2, len_residual_blocks - i)
            res_block_output = res_block_output.repeat((1, 1, spatial_repeat, spatial_repeat))
            skip_connections.append(res_block_output)
        x = self.bn_lrelu(x)
        return self.out_conv(x), skip_connections


class Decoder(nn.Module):
    def __init__(self, n_channels, input_channels, n=4, regular_skip=3, light_skip=1, max_channels=256, min_channels=4):
        super(Decoder, self).__init__()
        self.light_skip_con_count = light_skip
        self.regular_skip_con_count = regular_skip

        self.ups = nn.Sequential()
        self.skip_convs = nn.Sequential()

        num = 128
        up_count = 7
        for i in range(up_count):
            regular_skip_channel_count = np.power(2, up_count-i+1)
            skip_channels_count = self.light_skip_con_count*1 + self.regular_skip_con_count*min(max_channels, regular_skip_channel_count)
            skip_out_ch = max(10, regular_skip_channel_count)
            self.skip_convs.add_module(str(i), Inconv(skip_channels_count, skip_out_ch, kernel_size=1).to(device))
            if i == 0:
                in_ch = min(max_channels, n * num)*self.regular_skip_con_count + 256*self.light_skip_con_count + skip_out_ch
            else:
                in_ch = min(max_channels, n * num) + skip_out_ch
            out_ch = min(max_channels, n * num // 2)
            self.ups.add_module(str(i), up(in_ch, out_ch, inter_ch=max(min_channels, n * num // 4), conv_times=min(i+1, 2)).to(device))
            num = num // 2

        self.out_conv = Inconv(n * num, n_channels).to(device)

    def forward(self, x, skip_connections):
        len_ups = len(self.ups)
        for i in range(len_ups):
            skip = self.skip_convs[i](skip_connections[len_ups - i - 1])
            x = self.ups[i](x, skip)
        x = self.out_conv(x)
        return x

    def param_count(self):
        upconvs = np.sum(p.numel() for up in self.skip_convs for p in up.parameters() if p.requires_grad)
        ups = np.sum(p.numel() for up in self.ups for p in up.parameters() if p.requires_grad)
        others = np.sum(p.numel()  for p in self.parameters() if p.requires_grad)
        return ups+others+upconvs


class UNetRenderer(BaseModel):
    def __init__(self, light_types=17, h=128, w=128, encoded_dim=4, out_range=None):
        super(UNetRenderer, self).__init__('renderer')

        self.light_types = light_types
        # one encoder input for each of albedo, normal, depth, and light
        self.encoder_total_inputs = 4

        self.h = h
        self.w = w
        self.encoded_dim = encoded_dim

        channels = [3, 32, 64, 128, 256, 256, 256]
        kernel_size = 3
        padding = 1

        ## stride of 1 on first layer and 2 everywhere else
        stride_fn = lambda ind: 1 if ind == 0 else 2

        sys.stdout.write('<Renderer> Building albedo encoder:\t')
        self.albedo_encoder = build_encoder(channels, kernel_size, padding, stride_fn).to(device)
        sys.stdout.write('<Renderer> Building normal encoder:\t')
        self.normal_encoder = build_encoder(channels, kernel_size, padding, stride_fn).to(device)

        # set input channel = 1 for depth and light
        channels[0] = 1
        sys.stdout.write('<Renderer> Building depth encoder:\t')
        self.depth_encoder = build_encoder(channels, kernel_size, padding, stride_fn).to(device)

        sys.stdout.write('<Renderer> Building light encoder:\t')
        in_ch = self.light_types
        out_ch = self.light_types*2
        sys.stdout.write('%3d --> %3d --> %3d\n'%(in_ch, out_ch, out_ch))
        self.light_encoder = Inconv(in_ch, out_ch, kernel_size=1, conv_times=2).to(device)

        ## set output channel to 3 for renderer rgb output
        channels[0] = 3

        ## link encoder and decoder
        channels.append(channels[-1])

        ## reverse order for decoder
        channels = list(reversed(channels))
        channels.append(3)

        ## all encoded channels are concatenated
        ## channel count for albedo, depth, and normal
        channels[0] *= (self.encoder_total_inputs - 1)
        ## add self.light_types*2 channels for lighting
        channels[0] += self.light_types*2

        # build decoder
        self.upsampler = nn.UpsamplingNearest2d(scale_factor=2)
        stride_fn = lambda ind: 1
        sys.stdout.write('<Renderer> Building decoder:\t\t')
        self.decoder = build_encoder(channels, kernel_size, padding, stride_fn, mult=self.encoder_total_inputs).to(device)
        in_channels = (self.encoder_total_inputs) * channels[-2] - 2
        out_channels = channels[-1]
        self.decoder[-1] = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding).to(device)

        self.out_range = out_range
        if self.out_range is not None:
            self.out_range = list(map(int, out_range.split(",")))
            self.tanh = nn.Tanh()

    def encode(self, encoder, x):
        encoded = [x]
        for ind in range(len(encoder)):
            x = encoder[ind](x)
            x = F.leaky_relu(x)
            encoded.append(x)
        return x, encoded

    def decode(self, decoder, x, skips, upsample_skip_ind=[0, -1]):
        upsample_skip_ind = [x + len(decoder) - 1 if x < 0 else x for x in upsample_skip_ind]
        for ind in range(len(decoder) - 1):
            x = decoder[ind](x)
            if ind not in upsample_skip_ind:
                x = self.upsampler(x)
            x = join(1)(x, skips[-(ind + 1)])
            x = F.leaky_relu(x)

        return decoder[-1](x)

    def forward_patch(self, albedo, depth, normal, light_type):
        encoded_albedo, skip_connections_albedo = self.encode(self.albedo_encoder, albedo)
        encoded_depth, skip_connections_depth = self.encode(self.depth_encoder, depth)
        encoded_normal, skip_connections_normal = self.encode(self.normal_encoder, normal)

        light_type = light_type.reshape(light_type.shape[0], light_type.shape[1], 1, 1).repeat(
            (1, 1, encoded_albedo.shape[2], encoded_albedo.shape[3]))
        encoded_light = self.light_encoder(light_type)

        encoded_input = tuple(x for x in (encoded_albedo, encoded_depth, encoded_normal, encoded_light) if x is not None)
        encoded_input = th.cat(encoded_input, dim=1)
        skip_connections = []
        for i in range(len(skip_connections_albedo)):
            skips = tuple(x for x in (skip_connections_albedo[i], skip_connections_depth[i], skip_connections_normal[i]) if x is not None)
            skip_connections.append(th.cat(skips, dim=1))

        render = self.decode(self.decoder, encoded_input, skip_connections)
        if self.out_range is not None:
            render = self.tanh(render)
            render = (render + 1) * (self.out_range[1] - self.out_range[0]) / 2 + self.out_range[0]
        return render



    def forward(self, albedo, depth, normal, light_type, mask=None, mean_shift=True, return_errors=False):
        return self.forward_patch(albedo, depth, normal, light_type)


# from code.model.UNetRenderer2 import UNetRenderer as UNetRenderer2
# model2 = UNetRenderer2(out_range=out_range)
# model2.load_pretrained_model('../models/')
# i = 0
# model.albedo_encoder[i].conv.load_state_dict(model2.albedo_encoder[i].state_dict())
# model.normal_encoder[i].conv.load_state_dict(model2.normal_encoder[i].state_dict())
# model.depth_encoder[i].conv.load_state_dict(model2.depth_encoder[i].state_dict())
#
# for i in range(1, 4):
#     model.normal_encoder[i][0].conv.load_state_dict(model2.normal_encoder[i][0].conv.state_dict())
#     model.normal_encoder[i][1].conv.load_state_dict(model2.normal_encoder[i][1].state_dict())
#     model.depth_encoder[i][0].conv.load_state_dict(model2.depth_encoder[i][0].conv.state_dict())
#     model.depth_encoder[i][1].conv.load_state_dict(model2.depth_encoder[i][1].state_dict())
#     model.albedo_encoder[i][0].conv.load_state_dict(model2.albedo_encoder[i][0].conv.state_dict())
#     model.albedo_encoder[i][1].conv.load_state_dict(model2.albedo_encoder[i][1].state_dict())
#
# model.albedo_encoder[4].conv.load_state_dict(model2.albedo_encoder[4].state_dict())
# model.normal_encoder[4].conv.load_state_dict(model2.normal_encoder[4].state_dict())
# model.depth_encoder[4].conv.load_state_dict(model2.depth_encoder[4].state_dict())
# model.albedo_encoder[5].load_state_dict(model2.albedo_encoder[5].state_dict())
# model.normal_encoder[5].load_state_dict(model2.normal_encoder[5].state_dict())
# model.depth_encoder[5].load_state_dict(model2.depth_encoder[5].state_dict())
# model.light_encoder.load_state_dict(model2.light_encoder.state_dict())
#
# model.decoder[0].conv.load_state_dict(model2.decoder[0].state_dict())
# for i in range(1, 7):
#     model.decoder[i][0].conv.load_state_dict(model2.decoder[i][0].conv.state_dict())
#     model.decoder[i][1].conv.load_state_dict(model2.decoder[i][1].state_dict())
# model.decoder[7].load_state_dict(model2.decoder[7].state_dict())