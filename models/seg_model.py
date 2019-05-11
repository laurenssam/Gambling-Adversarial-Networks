"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import pylab as plt
from torchvision import transforms    
import scipy.misc
from sklearn.metrics import confusion_matrix
from .focal_loss import FocalLoss2d  


class SegModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults(dataset_mode='aligned')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        if is_train:
            parser.add_argument('--lambda_regression', type=float, default=1.0, help='weight for the regression loss')  # You can define new arguments for this model.
            parser.add_argument('--alpha', type=float, default=0.8, help='weight for the class unbalancing')
            parser.add_argument('--weighting', type=str, default="mean", help='type of class weighting (mf/class')
            parser.add_argument('--gamma', type=float, default=2, help='discount weight factor')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for the regression loss')  # You can define new arguments for this model.


        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['data_A', 'data_B', 'output']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['G']
        self.trans = transforms.ToPILImage()
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        if opt.class_name != None:
            opt.output_nc = 2
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)

            if opt.class_name != None:
                self.weight = torch.tensor([1.0, 200.0]).to(self.device)
            else:
                self.weight = torch.tensor([2.5, 24.1, 4.9, 209.9, 151.4, 86.8, 498.4, 186.2, 6.1200, 113.6, 17.2, 82.9, 632.9, 16.9, 446.,411.5, 450.2, 1158.9, 274.6]).to(self.device)
            # self.weight = torch.tensor([2.72, 16.5, 4.39, 143.95, 118.48, 82.47, 524.25, 145.44, 6.24, 67.98, 23.85, 88.5, 1040.56, 14.73, 761.47, 390.39, 457.47, 626.6, 318.81]).to(self.device)

            self.criterionLoss = torch.nn.CrossEntropyLoss(weight=self.weight**(opt.alpha), ignore_index=255)
            # self.criterionLoss = FocalLoss2d(weight=self.weight**(0.25), gamma = 2)
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, weight_decay=5e-4)
            self.optimizers = [self.optimizer_G]
        self.counter = 0
        self.epoch = 0
        if opt.pretrained == "full":
            state_dict = torch.load("latest_net_G_full.pth", map_location=self.device)
            self.netG.load_state_dict(state_dict)
        elif opt.pretrained == "split":
            state_dict = torch.load("latest_net_G_split.pth", map_location=self.device)
            self.netG.load_state_dict(state_dict)
        else:
            print("Starting from scratch")

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        self.data_A = input['A' if AtoB else 'B'].to(self.device)  # get image data A
        self.data_B = input['B' if AtoB else 'A'].to(self.device)  # get image data B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # get image paths

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.output = self.netG(self.data_A)  # generate output image given the input data_A


    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_G = self.criterionLoss(self.output, self.data_B.long()) * self.opt.lambda_regression
        self.loss_G.backward()       # calculate gradients of network G w.r.t. loss_G
        self.output = torch.argmax(self.output, dim=1)


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer_G.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer_G.step()        # update gradients for network G
