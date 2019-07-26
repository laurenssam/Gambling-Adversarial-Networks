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
from torchvision import transforms    
import scipy.misc
from sklearn.metrics import confusion_matrix
from .focal_loss import FocalLoss2d  
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt


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
            parser.add_argument('--gamma', type=float, default=2, help='discount weight factor')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for the regression loss')  # You can define new arguments for this model.
            parser.add_argument('--focal', type=int, default=0, help='weight for the class unbalancing')


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
        self.visual_names = ['real_A', 'real_B', 'fake_B_output']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['G']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)
        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            if self.opt.dataset == "voc":
                self.weight = torch.tensor([1.59, 78.8, 89., 81.72, 113.67, 144., 57.9, 39.8, 25.62, 72.51, 133.7, 96., 27.4, 83.63, 67.7,  11.2, 137.6, 119.2, 80.3, 59., 111.]).to(self.device)
            elif self.opt.dataset == "camvid":
                self.weight = torch.tensor([0.588, 0.510, 2.6966, 0.45, 1.17, 0.770, 2.47, 2.52, 1.01, 3.237, 4.131]).to(self.device)
            else:
                self.weight = torch.tensor([2.5, 24.1, 4.9, 209.9, 151.4, 86.8, 498.4, 186.2, 6.1200, 113.6, 17.2, 82.9, 632.9, 16.9, 446.,411.5, 450.2, 1158.9, 274.6]).to(self.device)
            if self.opt.focal:
                self.criterionLoss = FocalLoss2d(weight=self.weight**(0.25), gamma=self.opt.gamma)
                print(self.opt.gamma)
            else:
                self.criterionLoss = torch.nn.CrossEntropyLoss(weight=self.weight**(opt.alpha), ignore_index=255)
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=5e-4)
            self.optimizers = [self.optimizer_G]
        self.counter = 0
        self.epoch = 0
        if opt.pretrained == "full":
            filename = "latest_net_G_full_" + self.opt.dataset + "_" + self.opt.netG + ".pth" 
            state_dict = torch.load(filename, map_location=self.device)
            self.netG.load_state_dict(state_dict)
            print(filename + "loaded")
        else:
            print("Starting from scratch")
        print(self.netG)
        if not os.path.exists("checkpoints/" + opt.name + "/gradients"):
            os.mkdir("checkpoints/" + opt.name + "/gradients") 
        self.norms_G = []


        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.real_A = input['A'].to(self.device)  # get image data A
        self.real_B = input['B'].to(self.device)  # get image data B
        self.image_paths = input['A_paths']  # get image paths


    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # generate output image given the input data_A
        # print(self.isTrain)
        # print(self.netG.training)
        
        # print("Shape output: ", self.fake_B.shape)
        # print("-" * 20)

    def hist_gradient(self, parameters, name):
        counter = 0
        store = []
        for _ , p in parameters:
            if len(p.shape) > 3:
                store.append(("conv" + str(counter) + "_" + str(p.shape), p.flatten().cpu().data))
                counter += 1
        fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(15,15), constrained_layout=True)
        counter = 0
        if name == "G":
            img_path = "checkpoints/" + self.opt.name + "/gradients/histogram_iter" + str(self.counter) + "_" + name + ".png"
        else:
            img_path = "checkpoints/" + self.opt.name + "/gradients/histogram_iter" + str(self.iteration_D) + "_" + name + ".png"
        for row in ax:
            for col in row:
                if counter < len(store):
                    temp = col.hist(store[counter][1], bins=1000)
                    col.set_title(store[counter][0])
                    counter += 1
        plt.savefig(img_path, dpi=100)
        plt.close()

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_G = self.criterionLoss(self.fake_B, self.real_B.long()) * self.opt.lambda_regression
        self.loss_G.backward()       # calculate gradients of network G w.r.t. loss_G
        # if self.counter % 50 == 0:
        #     self.hist_gradient(self.netG.named_parameters(), "G")
        self.counter += 1 
        total_norm = 0
        # for p in self.netG.parameters():
        #     param_norm = p.grad.data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)
        # self.norms_G.append(float(total_norm))
    

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer_G.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer_G.step()        # update gradients for network G
        self.fake_B_output = torch.argmax(self.fake_B.clone().detach(), dim=1)
        self.fake_B_output[self.real_B == 255] = 255

        # plt.plot(self.norms_G)
        # plt.title("Gradient norm G")
        # plt.savefig("checkpoints/" + self.opt.name + "/norm_G.png")
        # plt.close()
