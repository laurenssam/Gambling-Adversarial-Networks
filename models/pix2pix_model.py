import torch
from .base_model import BaseModel
from . import networks
# from visualize import *
import numpy as np
import cityscapes
import scipy.misc as sc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from .focal_loss import FocalLoss2d  



class Pix2PixModel(BaseModel):
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
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--alpha', type=float, default=0.25, help='weight for the class unbalancing')
            parser.add_argument('--gamma', type=float, default=2., help='scale for the focal loss')
            parser.add_argument('--weighting', type=str, default="mean", help='type of class weighting (mf/class')
            # parser.add_argument('--num_D', type=int, default=1, help='Number of discriminators')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_CE']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B_output', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
       
        self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            for i in range(1, self.opt.num_D + 1):
                temp = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                setattr(self, "netD" + str(i), temp)
                print(temp)
                self.loss_names.extend(['D' + str(i) +'_fake', 'D' + str(i) +'_real'])
                setattr(self, 'loss_D' + str(i) +'_fake', 0)
                setattr(self, 'loss_D' + str(i) +'_real', 0)
                # self.optimizers.append(torch.optim.SGD(temp.parameters(), lr=opt.lr_D, momentum=0.9))
                self.optimizers.append(torch.optim.Adam(temp.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=1e-5))
                self.model_names.extend(['D' + str(i)])

            self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)


        if self.isTrain:
            self.ignore = 255
            # define loss functions
            if self.opt.dataset == "voc":
                self.weight = torch.tensor([1.59, 78.8, 89., 81.72, 113.67, 144., 57.9, 39.8, 25.62, 72.51, 133.7, 96., 27.4, 83.63, 67.7,  11.2, 137.6, 119.2, 80.3, 59., 111.]).to(self.device)
            elif self.opt.dataset == "camvid":
                self.weight = torch.tensor([0.588, 0.510, 2.6966, 0.45, 1.17, 0.770, 2.47, 2.52, 1.01, 3.237, 4.131]).to(self.device)
            else:
                self.weight = torch.tensor([2.5, 24.1, 4.9, 209.9, 151.4, 86.8, 498.4, 186.2, 6.1200, 113.6, 17.2, 82.9, 632.9, 16.9, 446.,411.5, 450.2, 1158.9, 274.6]).to(self.device)

            # self.criterionCE = FocalLoss2d(weight=self.weight**(opt.alpha), gamma=opt.gamma)
            self.criterionCE = torch.nn.CrossEntropyLoss(weight=self.weight**(opt.alpha), ignore_index=self.ignore)  
            # self.criterionCE = torch.nn.CrossEntropyLoss( ignore_index=self.ignore)  

            # self.criterionCE = FocalLoss2d(weight=self.weight**(0.25), gamma = 2)
          
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=5e-4)

            self.optimizers.append(self.optimizer_G)

            ## Training schedule
            self.count_D = 0
            self.count_G = 0
            self.pretrain = (opt.pretrain_D > 0)

            self.pretrain_D = opt.pretrain_D
            self.D_train = opt.D_train
            self.G_train = opt.G_train

            # Initialisation
            self.loss_G_GAN = 0
            self.loss_G_CE = 0
            if opt.pretrained == "full":
                filename = "latest_net_G_full_" + self.opt.dataset + "_" + self.opt.netG + ".pth" 
                state_dict = torch.load(filename, map_location=self.device)
                self.netG.load_state_dict(state_dict)
                print(filename + "loaded")
            elif opt.pretrained == "early":
                filename = "latest_net_G_early_" + self.opt.dataset + "_" + self.opt.netG + ".pth" 
                state_dict = torch.load(filename, map_location=self.device)
                self.netG.load_state_dict(state_dict)
                print(filename + "loaded")
            else:
                print("Starting from scratch")
   

            self.softmax = torch.nn.Softmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()
            self.accuracies = []
            self.epoch = 0
            self.norms_G = []
            self.norms_D = []
            self.mean_max = []

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
        self.fake_B = self.netG(self.real_A)  # G(A) ## Convolution output N X 19 X 512 X 512
        if self.netG.training:
            valid_region_mask = (self.real_B != self.ignore).unsqueeze(1).expand(self.fake_B.shape).float() ## N X 19 X 512 X 512 0 for where real is ignore label
            self.input_discr_fake = self.softmax(self.fake_B) * valid_region_mask ## Elementwise multiplication: Zeroing out the ignore label

    
    def to_one_hot(self, labels, C):
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        target = one_hot.scatter_(1, labels.cpu(), 1)
        # print(test[(test == labels.cpu())].flatten())
        # print(labels[(test == labels.cpu())].flatten())
        return target.cuda().float()


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        self.real_B_one_hot_temp = self.real_B.clone() ## cloning the real image
        self.real_B_one_hot_temp[self.real_B_one_hot_temp == self.ignore] = 0 ## Replacing ignore label by zeroes
        self.real_B_one_hot = self.to_one_hot(self.real_B_one_hot_temp.unsqueeze(dim=1), self.opt.output_nc) ## convert to one-hot
        valid_region_mask = (self.real_B != self.ignore).unsqueeze(1).expand(self.real_B_one_hot.shape).float() ## see forward(self)
        self.real_B_one_hot = self.real_B_one_hot * valid_region_mask

        self.fake_B = self.fake_B.detach()
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
            pred_fake = discriminator(fake_AB)
            loss_fake = self.criterionGAN(pred_fake, False)

            # Real
            real_AB = torch.cat((input_rgb, input_real), 1)
            pred_real = discriminator(real_AB)
            loss_real = self.criterionGAN(pred_real, True)

            # combine loss and calculate gradients
            self.loss_D = (loss_fake + loss_real) * 0.5
            self.loss_D.backward()
            self.optimizers[i-1].step()

            setattr(self, 'loss_D' + str(i) + '_fake', loss_fake.item())
            setattr(self, 'loss_D' + str(i) + '_real', loss_real.item())

            if i != self.opt.num_D:
                input_real, input_fake, input_rgb = self.downsample(input_real), self.downsample(input_fake), self.downsample(input_rgb)


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.loss_G_GAN = 0
        input_rgb = self.real_A
        input_fake = self.input_discr_fake
        self.optimizer_G.zero_grad()
        for i in range(1, self.opt.num_D + 1):
            discriminator = getattr(self, 'netD' + str(i))
            self.set_requires_grad(discriminator, False)  # disable backprop for D
            fake_AB = torch.cat((input_rgb, input_fake), 1)
            pred_fake = discriminator(fake_AB)
            self.loss_G_GAN += (self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN * (1/self.opt.num_D))
            if i < self.opt.num_D:
                input_fake, input_rgb = self.downsample(input_fake), self.downsample(input_rgb) 

        # Second, G(A) = B
        self.loss_G_CE = self.criterionCE(self.fake_B, self.real_B) 
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_CE + self.loss_G_GAN
        self.loss_G.backward()
        self.optimizer_G.step()



    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        if self.pretrain:
            self.backward_D()                # calculate gradients for D
            self.count_D += 1
            if self.count_D == self.pretrain_D:  
                self.pretrain = False
                self.count_D = 0
                self.count_G = 0
                print("Finished pretraining the discriminator")
        else:
            if self.count_G < self.G_train:
                self.backward_G()
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

