import torch
from .base_model import BaseModel
from . import networks
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# from visualize import *
import copy

class ELModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for Embedding loss')
            parser.add_argument('--alpha', type=float, default=0.25, help='weight for the class unbalancing')
            parser.add_argument('--weighting', type=str, default="mean", help='type of class weighting (mf/class')
            parser.add_argument('--loss_D', type=str, default="bce", help='type of loss for discriminator (bce/emb)')
            parser.add_argument('--start_adv', type=int, default=0, help='epoch number that adversarial training starts')


        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['EL', 'CE']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B_output', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            for i in range(1, self.opt.num_D + 1):
                temp = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                setattr(self, "netD"+str(i), temp)
                self.loss_names.extend(['D' + str(i) +'_fake', 'D' + str(i) +'_real'])
                setattr(self, 'loss_D' + str(i) +'_fake', 0)
                setattr(self, 'loss_D' + str(i) +'_real', 0)
                # self.optimizers.append(torch.optim.SGD(temp.parameters(), lr=opt.lr_D, momentum=0.9))
                self.optimizers.append(torch.optim.Adam(temp.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=1e-5))
                self.model_names.extend(['D' + str(i)])

            self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
            # Number of pretrain iterations
            self.pretrain_G = opt.pretrain_G
            self.count_G = 0

            self.pretrain_D = opt.pretrain_D
            self.count_D = 0
            self.pretrain = self.pretrain_D > 0

            # Initalize the loss functions for plot
            self.loss_EL = 0
            self.loss_CE = 0
            # Number of steps the generator and discriminator train in a row 
            self.G_train = opt.G_train
            self.D_train = opt.D_train

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.ignore = 255
            # define loss functions
            if self.opt.dataset == "voc":
                self.weight = torch.tensor([1.59, 78.8, 89., 81.72, 113.67, 144., 57.9, 39.8, 25.62, 72.51, 133.7, 96., 27.4, 83.63, 67.7,  11.2, 137.6, 119.2, 80.3, 59., 111.]).to(self.device)
            elif self.opt.dataset == "camvid":
                self.weight = torch.tensor([0.588, 0.510, 2.6966, 0.45, 1.17, 0.770, 2.47, 2.52, 1.01, 3.237, 4.131]).to(self.device)
            else:
                self.weight = torch.tensor([2.5, 24.1, 4.9, 209.9, 151.4, 86.8, 498.4, 186.2, 6.1200, 113.6, 17.2, 82.9, 632.9, 16.9, 446.,411.5, 450.2, 1158.9, 274.6]).to(self.device)

            self.criterionCE = torch.nn.CrossEntropyLoss(weight=self.weight**(opt.alpha), ignore_index=self.ignore)            
            self.criterionEL = torch.nn.MSELoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=5e-4)
            self.optimizers.append(self.optimizer_G)

            if opt.pretrained == "full":
                filename = "latest_net_G_full_" + self.opt.dataset + "_" + self.opt.netG + ".pth" 
                state_dict = torch.load(filename, map_location=self.device)
                self.netG.load_state_dict(state_dict)
                print(filename + "loaded")
            elif opt.pretrained == "early":
                filename = "latest_net_G_early_" + self.opt.dataset + "_" + self.opt.netG + ".pth" 
                state_dict = torch.load(filename, map_location=self.device)
                self.netG.load_state_dict(state_dict)

            self.softmax = torch.nn.Softmax(dim=1)

            # Statistics of discriminator
            self.accuracies = []
            self.sigmoid = torch.nn.Sigmoid()
            self.correct_predictions = 0

            # counters
            self.counter_stats = 0

            self.features = {}
            self.epoch = 0
            self.norms_G = []
            self.norms_D = []
            self.mean_max = []
            self.losses_real = []
            self.losses_fake = []
            self.embedding_losses = []
            self.CE_losses = []
            self.iteration_D = 0
            self.iteration_G = 0

            # self.pretrain_D = 0

    def to_one_hot(self, labels, C):
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        target = one_hot.scatter_(1, labels.cpu(), 1)
        return target.to(self.device).float()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG(self.real_A)  # G(A)

        # valid_region_mask = (self.real_B != self.ignore).unsqueeze(1).expand_as(self.fake_B).float()
        # self.input_discr_fake = self.softmax(self.fake_B.clone()) * valid_region_mask
        self.fake_B = self.netG(self.real_A)  # G(A) # Forward pass
        if self.netG.training:
            self.mask = (self.real_B != self.ignore).unsqueeze(dim=1).expand(self.fake_B.shape).float() # mask for ignore class, zeroes out the ignore class    
            # max_indices = self.to_one_hot(torch.argmax(self.fake_B.detach(), dim=1).unsqueeze(dim=1), 19) * 5 # N x 512 X 512 --> N X 1 X 512 X 512 --> N X 19 X 512 X 512 --> * 100
            # self.input_discr_fake = self.softmax(self.fake_B.clone() + max_indices) * self.mask # fake * masking
            self.input_discr_fake = self.softmax(self.fake_B) * self.mask - (1-self.mask)

            self.real_B_one_hot_temp = self.real_B.clone() ## cloning the real image
            self.real_B_one_hot_temp[self.real_B_one_hot_temp == self.ignore] = 0 ## 
            self.real_B_one_hot = self.to_one_hot(self.real_B_one_hot_temp.unsqueeze(dim=1), self.opt.output_nc)
            self.real_B_one_hot = self.real_B_one_hot * self.mask - (1-self.mask)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # self.real_B_one_hot_temp = self.real_B.clone() ## cloning the real image
        # self.real_B_one_hot_temp[self.real_B_one_hot_temp == self.ignore] = 0 ## Replacing ignore label by zeroes
        # self.real_B_one_hot = self.to_one_hot(self.real_B_one_hot_temp.unsqueeze(dim=1), 19) ## convert to one-hot
        # valid_region_mask = (self.real_B != self.ignore).unsqueeze(1).expand_as(self.fake_B).float() ## see forward(self)
        # self.real_B_one_hot = self.real_B_one_hot * valid_region_mask

        input_real = self.real_B_one_hot
        input_fake = self.input_discr_fake.detach()
        input_rgb = self.real_A
        for i in range(1, self.opt.num_D + 1):
            # sc.imsave("rgb" + str(i) + ".png", input_rgb[0].permute(1, 2, 0).cpu().numpy())
            # fake = cityscapes.visualize(cityscapes.colorize_mask(torch.argmax(input_fake[0], dim=0).detach().cpu().numpy()).convert("RGB")).permute(1, 2, 0)
            # print(fake.shape)
            # sc.imsave( "fake" + str(i) + ".png", fake)
            # sc.imsave( "real" + str(i) + ".png", cityscapes.visualize(cityscapes.colorize_mask(torch.argmax(input_real[0], dim=0).detach().cpu().numpy()).convert("RGB")).permute(1, 2, 0))
            discriminator = getattr(self, 'netD' + str(i))
            self.set_requires_grad(discriminator, True)  # enable backprop for D
            self.optimizers[i-1].zero_grad()

            fake_AB = torch.cat((input_rgb, input_fake), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = discriminator(fake_AB.detach())
            # correct_predictions = (self.sigmoid(pred_fake) < 0.5).float().sum()
            loss_fake = self.criterionGAN(pred_fake, False)
            setattr(self, 'loss_D' + str(i) + '_fake', loss_fake)

            # Real
            real_AB = torch.cat((input_rgb, input_real), 1)
            pred_real = discriminator(real_AB)
            # correct_predictions += (self.sigmoid(pred_real) > 0.5).float().sum()
            loss_real = self.criterionGAN(pred_real, True)
            setattr(self, 'loss_D' + str(i) + '_real', loss_real)
            # self.accuracies.append(float(correct_predictions/(np.prod(pred_fake.shape) * 2)))


            # combine loss and calculate gradients
            self.loss_D = (loss_fake + loss_real) * 0.5
            self.loss_D.backward()
            self.optimizers[i-1].step()
            if self.iteration_D % 50:
                self.losses_real.append(loss_real.item())
                self.losses_fake.append(loss_fake.item())
            if i != self.opt.num_D:
                input_real, input_fake, input_rgb = self.downsample(input_real), self.downsample(input_fake), self.downsample(input_rgb)
        self.iteration_D += 1

    def backward_G(self):
        """Calculate embedding loss and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
                # combine loss and calculate gradients

        input_real = self.real_B_one_hot
        input_fake = self.softmax(self.fake_B) * self.mask - (1-self.mask)
        input_rgb = self.real_A
        self.loss_EL = 0

        for i in range(1, self.opt.num_D + 1):
            discriminator = getattr(self, 'netD' + str(i))
            self.set_requires_grad(discriminator, False)  # disable backprop for D

            fake_AB = torch.cat((input_rgb, input_fake), 1)
            pred_fake = discriminator(fake_AB, emb=True)

            with torch.no_grad():
                real_AB = torch.cat((input_rgb, input_real), 1)
                pred_real = discriminator(real_AB, emb=True)
            for j, emb_fake in enumerate(pred_fake):
                self.loss_EL += (self.criterionEL(emb_fake, pred_real[j].detach()) * self.opt.lambda_GAN * (1/self.opt.num_D))
            if i != self.opt.num_D:
                input_real, input_fake, input_rgb = self.downsample(input_real), self.downsample(input_fake), self.downsample(input_rgb)
        self.loss_CE = self.criterionCE(self.fake_B, self.real_B)
        if self.epoch > self.opt.start_adv:
            self.loss_G = self.loss_EL + self.loss_CE
        else:
            self.loss_G = self.loss_CE
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.loss_G.backward()
        self.optimizer_G.step()             # udpate G's weights
        if self.iteration_G % 50 == 0:
            self.embedding_losses.append(self.loss_EL.item())
            self.CE_losses.append(self.loss_CE.item())
        self.iteration_G += 1


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        if self.pretrain:
            if self.count_D < self.pretrain_D:
                self.backward_D()
                self.count_D += 1 
            else:
                print("Finished the pretraining")
                self.pretrain = False
                self.count_D = 0
                self.count_G = 0
        else:
            if self.count_G < self.G_train:
                self.backward_G()                   # calculate graidents for G
                self.count_G += 1  
                if self.count_G == self.G_train:
                    print("Finished schedule G: evaluation mode")
            elif self.count_D < self.D_train:
                self.backward_D()
                self.count_D += 1
                if self.count_D == self.D_train:
                    self.count_D = 0
                    self.count_G = 0
                    print("Finished schedule D: training mode")
        self.fake_B_output = torch.argmax(self.fake_B.detach(), dim=1)   ## N X 512 x 512
        self.fake_B_output[self.real_B == self.ignore] = self.real_B[self.real_B == self.ignore]
        if (self.iteration_G + self.iteration_D) %  200:
            plt.plot(self.losses_fake)
            plt.plot(self.losses_real)
            plt.legend(["Loss_fake", "Loss_real"])
            plt.savefig("checkpoints/" + self.opt.name + "/train_loss_D.png")
            plt.close()

            plt.plot(self.embedding_losses)
            plt.plot(self.CE_losses)
            plt.legend(["emb_loss", "CE_loss"])
            plt.savefig("checkpoints/" + self.opt.name + "/train_loss_G.png")
            plt.close()




    # def calculate_stats(self, embs):
    #     names = ["emb_fake2", "emb_real2", "emb_fake4", "emb_real4"]
    #     for i, emb in enumerate(embs):
    #         shape = emb.shape
    #         emb = emb[0].reshape(1, shape[1], -1).squeeze()
    #         mean_channel = torch.mean(emb, dim=1).detach().cpu().numpy()
    #         max_channel = torch.max(emb, dim=1)[0].detach().cpu().numpy()
    #         min_channel = torch.min(emb, dim=1)[0].detach().cpu().numpy()
    #         f, (ax1, ax2, ax3) = plt.subplots(3)
    #         ax1.bar(np.arange(len(mean_channel)), mean_channel)
    #         ax1.set_title("Mean per channel")
    #         ax2.bar(np.arange(len(max_channel)), max_channel)
    #         ax2.set_title("Max per channel")
    #         ax3.bar(np.arange(len(min_channel)), min_channel)
    #         ax3.set_title("Min per channel")
    #         plt.savefig(str(self.counter_stats) + "_" + names[i])
    #         plt.close()
    #     self.counter_stats += 1
    #     return mean_channel, max_channel, min_channel