import numpy as np
from codes.model.util import Stich, join, normalize
from codes.model.unet_parts import *
from codes.model.base_model import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LightDecoder(nn.Module):
    def __init__(self, h=64, w=128, n=2, channels=None, light_type_count=17, min_channels=4, conv_times=2):
        super(LightDecoder, self).__init__()
        self.h = h
        self.w = w

        self.c = n * 128
        self.kernel_size = 1

        self.inter_ch = n * 64
        self.in_conv = Inconv(self.c, self.inter_ch, kernel_size=self.kernel_size, conv_times=conv_times).to(device)

        self.blocks = nn.Sequential()
        num_blocks = len(channels) - 2 if channels is not None else 7
        for i in range(num_blocks):
            if channels is not None:
                skip_channels = max(min_channels, channels[i + 1]) if i < num_blocks - 1 else 3
            else:
                skip_channels = max(min_channels, n * np.power(2, i))
            self.blocks.add_module(str(i), ResConv(self.inter_ch + skip_channels, self.inter_ch, kernel_size=self.kernel_size,
                                                   conv_times=conv_times).to(device))
        self.lrelu = nn.LeakyReLU(0.01)


        ####################################################################################
        # for reshaping light output
        self.encoded_dim = 4
        self.light_type_count = light_type_count
        self.reshaped_1x1_channels = self.inter_ch * (self.encoded_dim*self.encoded_dim)
        if conv_times == 1:
            self.out_conv = Conv(self.reshaped_1x1_channels, self.h * self.w, kernel_size=self.kernel_size).to(device)
            self.classification = Conv(self.reshaped_1x1_channels, self.light_type_count, kernel_size=self.kernel_size).to(device)
        else:
            self.out_conv = DoubleConv(self.reshaped_1x1_channels, self.h * self.w, inter_ch=self.light_type_count, kernel_size=self.kernel_size).to(device)
            self.classification = DoubleConv(self.reshaped_1x1_channels, self.light_type_count, kernel_size=self.kernel_size).to(device)
        self.pools = None

    def forward(self, x, skip_connections):

        if self.pools is None:
            self.pools = nn.Sequential()
            for i, tensor in enumerate(skip_connections[::-1]):
                self.pools.add_module(str(i), nn.MaxPool2d((int(tensor.shape[2] // x.shape[2]), int(tensor.shape[3] // x.shape[3]))))

        light = self.in_conv(x)
        block_len = len(self.blocks)
        for i in range(block_len):
            light = light + self.blocks[i](torch.cat((light, self.pools[i](skip_connections[block_len - i - 1])), dim=1))
        light = self.lrelu(light)

        #########################################################
        # reshape light
        _x = x.shape[2]//self.encoded_dim
        _y = x.shape[3]//self.encoded_dim
        light = light.view((light.shape[0], light.shape[1] * light.shape[2] * light.shape[3]  // (_x*_y), _x, _y))
        light_type = self.classification(light)
        light = self.out_conv(light)
        if light.shape[2] == 1 and light.shape[3] == 1:
            light = torch.reshape(light, (light.shape[0], 1, self.h, self.w))
            light_type = light_type.view(light_type.shape[0], light_type.shape[1])
        else:
            l = torch.zeros((light.shape[0], 1, light.shape[2]*self.h, light.shape[3]*self.w)).to(device)
            for i in range(light.shape[2]):
                for j in range(light.shape[3]):
                    l[:, :, self.h*i:self.h*(i+1), self.w*j:self.w*(j+1)] = torch.reshape(light[:,:,i,j], (light.shape[0], 1, self.h, self.w))
            light = l

        return light, light_type

class UNetDecomposer(BaseModel):
    def __init__(self, light_type_count=17, light_h=64, light_w=128,
                 h=128, w=128, out_range=None, output_last_ft=False):
        super(UNetDecomposer, self).__init__('decomposer')

        self.light_type_count = light_type_count
        self.light_h = light_h
        self.light_w = light_w
        self.h = h
        self.w = w
        self.output_last_ft = output_last_ft

        channels = [3, 32, 64, 128, 256, 256, 256]
        kernel_size = 3
        padding = 1

        ## stride of 1 on first layer and 2 everywhere else
        stride_fn = lambda ind: 1 if ind == 0 else 2
        sys.stdout.write('<Decomposer> Building encoder:\t\t')
        self.encoder = build_encoder(channels, kernel_size, padding, stride_fn).to(device)

        #######################
        #### decoders ####
        #######################

        sys.stdout.write('<Decomposer> Building normal decoder:\t')
        stride_fn = lambda ind: 1

        ## link encoder and decoder
        channels.append(channels[-1])

        ## reverse channel order for decoder
        channels = list(reversed(channels))

        ## separate albedo, normal, depth, and light decoders.
        ## mult = 2 because the skip layer concatenates
        ## an encoder layer with the decoder layer,
        ## so the number of input channels in each layer is doubled.
        self.decoder_normals = build_encoder(channels, kernel_size, padding, stride_fn, mult=2).to(device)

        sys.stdout.write('<Decomposer> Building albedo decoder:\t')
        channels.append(3)
        self.decoder_albedo = build_encoder(channels, kernel_size, padding, stride_fn, mult=2, conv_times=1).to(device)

        sys.stdout.write('<Decomposer> Building depth decoder:\t')
        channels[-1] = 1
        self.decoder_depth = build_encoder(channels, kernel_size, padding, stride_fn, mult=2).to(device)

        sys.stdout.write('<Decomposer> Building light decoder:\t\n')
        self.decoder_light = LightDecoder(h=64, w=128, channels=channels, light_type_count=17, min_channels=4,
                                          conv_times=2).to(device)

        self.upsampler = nn.UpsamplingNearest2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

        self.out_range = out_range
        if out_range is not None:
            self.out_range = list(map(int, out_range.split(",")))
            self.tanh = nn.Tanh()


    def __decode(self, decoder, encoded, inp, upsample_skip_ind=[0, -1], inp_shape=None):
        upsample_skip_ind = [x + len(decoder) - 1 if x < 0 else x for x in upsample_skip_ind]
        x = inp
        for ind in range(len(decoder) - 1):
            x = decoder[ind](x)
            if x.shape[2] == inp_shape[2]/2 and self.output_last_ft:
                last_ft = x
            if ind not in upsample_skip_ind:
                x = self.upsampler(x)
            x = join(1)(x, encoded[-(ind + 1)])
            x = F.leaky_relu(x)
        if x.shape[2] == inp_shape[2]/2 and self.output_last_ft:
            last_ft = x

        x = decoder[-1](x)
        if self.output_last_ft:
            return x, last_ft
        return x

    def get_pred_names(self):
        if self.albedo and self.depth and self.normal and self.light:
            return ['albedo', 'depth', 'normal', 'light', 'light_type']
        elif self.albedo and self.depth and self.normal and not self.light:
            return ['albedo', 'depth', 'normal', '_', '_']
        elif not self.albedo and self.depth and not self.normal and not self.light:
            return ['_', 'depth', '_', '_', '_']
        else:
            raise NotImplementedError("Do not recognize decomposer mode")


    def forward_patch(self, x):
        inp = x
        ## shared encoder
        encoded = [x]
        for ind in range(len(self.encoder)):
            x = self.encoder[ind](x)
            x = F.leaky_relu(x)
            encoded.append(x)

        ###################################################################
        ## light prediction
        light, light_type = self.decoder_light(x, encoded)

        ###################################################################
        ## albedo prediction
        albedo = self.__decode(self.decoder_albedo, encoded, x, inp_shape=inp.shape)
        albedo, albedo_feat = albedo if self.output_last_ft else [albedo, None]

        if self.out_range is not None:
            albedo = self.tanh(albedo)
            albedo = (albedo + 1)*(self.out_range[1] - self.out_range[0])/2 + self.out_range[0]

        ###################################################################
        ## normal prediction
        normals = self.__decode(self.decoder_normals, encoded, x, upsample_skip_ind=[0], inp_shape=inp.shape)
        normals, normals_feat = normals if self.output_last_ft else [normals, None]

        if self.out_range is not None:
            normals = self.tanh(normals)
            normals = (normals + 1) * (self.out_range[1] - self.out_range[0]) / 2 + self.out_range[0]

        ## G, R in [-1,1]
        gr = normals[:, 1:, :, :]*2 - 1
        # gr = torch.clamp(normals[:, 1:, :, :], -1, 1)
        ## B in [0,1]
        b = normals[:, [0], :, :]
        # b = torch.clamp(normals[:, 0, :, :].unsqueeze(1), 0, 1)
        clamped = torch.cat((b, gr), 1)
        ## normals are unit vector
        normals = normalize(clamped)

        ###################################################################
        ## depth prediction
        depth = self.__decode(self.decoder_depth, encoded, x, inp_shape=inp.shape)
        depth, depth_feat = depth if self.output_last_ft else [depth, None]

        if self.out_range is not None:
            depth = self.tanh(depth)
            depth = (depth + 1) * (self.out_range[1] - self.out_range[0]) / 2 + self.out_range[0]

        if self.output_last_ft:
            return (albedo, depth, normals, light, light_type), (albedo_feat, depth_feat, normals_feat, light_type)

        return albedo, depth, normals, light, light_type

    def get_patch(self, tensor, i, j):
        return tensor[:, :, j:j + self.h, i:i + self.w]

    def forward(self, x, mask=None, mean_shift=True):
        if x.shape[2] == self.w and x.shape[3] == self.h:
            return self.forward_patch(x)
        else:
            if mask is None:
                print("Mask is None. Processing images in fully convolutional way.")
                return self.forward_patch(x)

            # stitch patches to make full shoe
            # done when full-sized images are passed through
            input = x
            device = x.device

            # process all patches
            patch_interval = 1/2
            i_list = list(range(0, int(input.shape[3] - self.w) + 1, int(self.w * patch_interval)))
            # add last column in case it is not included in i_list
            if int(input.shape[3] - self.w) not in i_list:
                i_list.append(int(input.shape[3] - self.w))
            j_list = list(range(0, int(input.shape[2] - self.h) + 1, int(self.h * patch_interval)))
            # add last row in case it is not included in j_list
            if int(input.shape[2] - self.h) not in j_list:
                j_list.append(int(input.shape[2] - self.h))
            input_patches = torch.cat(tuple(self.get_patch(input, i, j) for i in i_list for j in j_list))
            output, features = self.forward_patch(input_patches)
            albedo_patches, depth_patches, normal_patches, light_patches, light_type_patches = output[0], output[1], output[2], output[3], output[4]

            # stich albedo patches
            shape_3d = (input.shape[0], 3, input.shape[2], input.shape[3])
            albedo = Stich(shape_3d, self.w, self.h, device, 0, 1, self.get_patch)
            albedo.stich(i_list, j_list, albedo_patches, mask, patch_interval, device, mean_shift=mean_shift)

            # stich depth patches
            shape_1d = (input.shape[0], 1, input.shape[2], input.shape[3])
            depth = Stich(shape_1d, self.w, self.h, device, 0, 1, self.get_patch)
            depth.stich(i_list, j_list, depth_patches, mask, patch_interval, device, mean_shift=mean_shift)

            # stich normal patches
            normal = Stich(shape_3d, self.w, self.h, device, -1, 1, self.get_patch)
            normal.stich(i_list, j_list, normal_patches, mask, patch_interval, device, mean_shift=mean_shift)

            if self.output_last_ft:
                return (albedo.image, depth.image, normal.image, None, None), (None, None, None, None, None)
            else:
                return albedo.image, depth.image, normal.image, None, None





# from codes.model.UNetDecomposer2 import UNetDecomposer as UNetDecomposer2
# model2 = UNetDecomposer2(albedo=True, depth=True, normal=True, light=True, out_range=out_range, output_last_ft=output_last_ft)
# model2.load_pretrained_model('../models/')
# model.encoder[0].conv.load_state_dict(model2.encoder[0].state_dict())
# for i in range(1, 4):
#     model.encoder[i][0].conv.load_state_dict(model2.encoder[i][0].conv.state_dict())
#     model.encoder[i][1].conv.load_state_dict(model2.encoder[i][1].state_dict())
# model.encoder[4].conv.load_state_dict(model2.encoder[4].state_dict())
# model.encoder[5].load_state_dict(model2.encoder[5].state_dict())
#
# for i in range(7):
#     model.decoder_albedo[i].conv.load_state_dict(model2.decoder_albedo[i].state_dict())
# model.decoder_albedo[7].load_state_dict(model2.decoder_albedo[7].state_dict())
#
# for i in range(3):
#     model.decoder_normals[i].conv.load_state_dict(model2.decoder_normals[i].state_dict())
#     model.decoder_depth[i].conv.load_state_dict(model2.decoder_depth[i].state_dict())
# for i in range(3, 6):
#     model.decoder_normals[i][0].conv.load_state_dict(model2.decoder_normals[i][0].conv.state_dict())
#     model.decoder_normals[i][1].conv.load_state_dict(model2.decoder_normals[i][1].state_dict())
#     model.decoder_depth[i][0].conv.load_state_dict(model2.decoder_depth[i][0].conv.state_dict())
#     model.decoder_depth[i][1].conv.load_state_dict(model2.decoder_depth[i][1].state_dict())
# model.decoder_normals[6].load_state_dict(model2.decoder_normals[6].state_dict())
# model.decoder_depth[6][0].load_state_dict(model2.decoder_depth[6][0].state_dict())
# model.decoder_depth[6][1].conv.load_state_dict(model2.decoder_depth[6][1].state_dict())
# model.decoder_depth[7].load_state_dict(model2.decoder_depth[7].state_dict())
#
# model.decoder_light.load_state_dict(model2.decoder_light.state_dict())
# model.save_checkpoint('../models/', None, 0, 0, 0, is_best=True)