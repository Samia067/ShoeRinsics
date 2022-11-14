import argparse
import os


class Options():
    def __init__(self, train=False):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.train = train
        self.initialized = False

    def initialize(self):

        self.parser.add_argument("--output", default="../results", help="Folder to save results and models.")
        self.parser.add_argument("--exp_name", default="shoerinsics", help="Name of experiment.")
        self.parser.add_argument("--weights_decomposer", required=self.train, default=None, help="Path to pretrained decomposer.")

        self.parser.add_argument("--dataroot", required=True, help="Root directory for datasets.")
        self.parser.add_argument("--val_dataset_dir", default=None, help="Directory for validation dataset.")

        self.parser.add_argument("--full", action='store_true', help="whether or not to load full images instead of patches")
        self.parser.add_argument("--patch", action='store_false', dest='full', help="whether or not to load patches instead of full images")
        self.parser.set_defaults(full=True)

        self.parser.add_argument("--test_time_aug", action='store_true', help="whether or not to use test time augmentation.")
        self.parser.set_defaults(test_time_aug=False)



        self.parser.add_argument("--num_workers", default=0, type=int, help="Number of threads for multiprocessing. Leave as 0 for a single process execution.")

        # training options
        self.parser.add_argument("--weights_renderer", default=None, help="Path to pretrained renderer. This must be provided for reconstruction visualizations.")
        self.parser.add_argument("--batch", default=8, type=int)
        self.parser.add_argument("--lr", default=1e-4, type=float)
        self.parser.add_argument("--train_renderer", action='store_true', help="whether or not to train the renderer.")
        self.parser.set_defaults(train_renderer=False)
        self.parser.add_argument("--train_net", action='store_true', help="whether or not to train the decomposer.")
        self.parser.set_defaults(train_net=False)
        self.parser.add_argument("--train_discriminator", action='store_true', help="whether or not to train the decomposer.")
        self.parser.set_defaults(train_discriminator=False)

        self.parser.add_argument("--weights_translator", default=None, help="Path to pretrained translator. The translator is trained as described in CycleGAN.")
        self.parser.add_argument("--weights_discriminator", default=None, help="Path to pretrained feature discriminator.")
        self.parser.add_argument("--syn_train_dataset_dir", default=None, help="Directory for synthetic training dataset.")
        self.parser.add_argument("--real_train_dataset_dir", default=None, help="Directory for real training dataset.")
        self.parser.add_argument("--syn_weights", default='1,2,1,0,0.03', help="Weights for synthetic training.")
        self.parser.add_argument("--real_weights", default='2,2', help="Weights for synthetic training.")

        # @click.option('--dataset', required=True, multiple=True)
        # self.parser.add_argument("--dataset", required=True, help="")

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()


        ######################################################

        # assert (opt.model in ['multimodal_single_model_supcon', 'multimodal_single_model', 'multimodal_single_model_supcon_normal',
        #                       'multimodal_single_model_normal', 'multimodal_two_models', 'supcon', 'multimodal_single_model_supcon_low_dim_normal'])
        #
        #
        # assert (opt.val in ['retrieval', 'triplets'])
        # assert (opt.test_ref in ['real_test_shoes', 'FID300', 'FID_matches', 'crimescene_data'])
        # assert (opt.test_crime in ['FID300_tracks'])
        # assert (opt.data in ['single', 'paired'])
        # assert (opt.input2 in [None, 'print'])
        # if opt.input2 == 'print':
        #     opt.input2 = [opt.input2]
        # if opt.data == 'paired':
        #     opt.batch = opt.batch // 2
        # # opt.input = sorted(opt.input.split(','))
        # opt.input = opt.input.split(',')
        # for i in opt.input:
        #     assert (i in ['rgb','print','depth'])
        #
        # if opt.model in ['multimodal_single_model', 'multimodal_single_model_normal']:
        #     assert('print' not in opt.input[1:])
        #
        # opt.occlusion_percents = sorted(list(map(int, opt.occlusion_percents.split(','))))
        # # opt.mask_percent = int(opt.mask_percent)
        # opt.occlusion_types = opt.occlusion_types.split(',')
        # for i in opt.occlusion_types:
        #     assert (i in ['right-left','left-right','top-bottom','bottom-top'])
        # # assert (opt.mask_occlusion_type in ['right-left', 'left-right', 'top-bottom', 'bottom-top'])
        # opt.occlusion_percent = 0
        # opt.occlusion_type = 'right-left'
        #
        # # opt.exp_types = opt.exp_types.split(',')
        # # for i in opt.exp_types:
        # #     assert (i in ['default','mask_database_image','mask_database_feature'])
        # assert (opt.exp_type in ['default', 'mask_database_image', 'mask_database_feature', 'mask_database_feature_not_query_feature'])
        # opt.mask_features = False
        #
        # if opt.warm:
        #     opt.warmup_from = 0.01
        #     opt.warmup_to = opt.lr
        #     # opt.warm_epochs = 10
        #     # if opt.cosine:
        #     #     eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        #     #     opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
        #     #             1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        #     # else:
        #     #     opt.warmup_to = opt.learning_rate

        # opt = setup(opt)
        # opt.exp_name = opt.output.rsplit("/", 1)[1]
        # opt.syn_only = opt.real_only = False
        # opt.max_dataset_size = np.inf


        args = vars(opt)
        opt.output = os.path.join(opt.output, opt.exp_name)
        opt.syn_weights = list(map(float, opt.syn_weights.split(",")))
        opt.real_weights = list(map(float, opt.real_weights.split(",")))  # [3, 2]

        # opt.syn_only = opt.real_only = False

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        os.makedirs(opt.output, exist_ok=True)
        file_name = os.path.join(opt.output, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        return opt