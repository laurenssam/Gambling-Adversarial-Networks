import torch
from .base_model import BaseModel
from . import networks
import torchvision
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from visualize import *
import numpy as np
import os
import os.path
import copy
from .focal_loss import FocalLoss2d  
import scipy.misc
import scipy.ndimage
import torch
import gc
import torchvision
import torchvision.models
from torch.utils import model_zoo
import pickle



class ARGModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='unet_512', netD='unet', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--alpha', type=float, default=0.25, help='Coefficient for weighting')
            parser.add_argument('--smooth', type=float, default=0.02, help='smoothing for the betting map')
            parser.add_argument('--input', type=int, default=0, help='only input the rgb as input')
            parser.add_argument('--peaked', type=int, default=0, help='only input the rgb as input')
            parser.add_argument('--zero_prediction', type=int, default=0, help='train on both the rgb only and rgb + prediction')
            parser.add_argument('--interleave', type=int, default=0, help='train on the ce map')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_fake', 'G_CE', 'real', 'fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_B_output', 'bet_map', 'bet_map2']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.ignore = 255
        self.n_classes = 19
        self.opt = opt
        output_nc = 1
        opt.init_type = "xavier"
        opt.init_gain = 0.25
        self.opt.netD = "unet"
        self.netD = networks.define_D(opt.input_nc + 1, opt.ndf, opt.netD,
                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, output_nc)

        self.BCEweight = torch.tensor([self.opt.weight_gambler]).to(self.device)
        self.criterionBCE = torch.nn.BCEWithLogitsLoss(pos_weight=self.BCEweight, reduction="none")
        self.criterionmatch = torch.nn.BCELoss()
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)


        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        print(self.netD)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            # state_dict = torch.load("latest_net_D.pth", map_location=self.device)
            # self.netD.load_state_dict(state_dict)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=5e-4)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # Initalize loss functions
            self.weight = torch.tensor([2.5, 24.1, 4.9, 209.9, 151.4, 86.8, 498.4, 186.2, 6.12, 113.6, 17.2, 82.9, 632.9, 16.9, 446.,411.5, 450.2, 1158.9, 274.6]).to(self.device)
            self.criterionCE = torch.nn.CrossEntropyLoss(weight=self.weight**(self.opt.alpha), ignore_index=self.ignore) 
            # self.criterionCE = FocalLoss2d(weight=self.weight**(0.25), gamma = 2)

            # layers for output U-nets

            # Loading the generator        
            if opt.pretrained == "full":
                state_dict = torch.load("latest_net_G_full.pth", map_location=self.device)
                self.netG.load_state_dict(state_dict)
            elif opt.pretrained == "split":
                state_dict = torch.load("latest_net_G_split.pth", map_location=self.device)
                self.netG.load_state_dict(state_dict)
            else:
                print("Starting from scratch")

            # stats
            self.epoch = 0
            self.count_D = 0
            self.count_G = 0
            self.D_train = opt.D_train
            self.G_train = opt.G_train
            self.pretrain_D = opt.pretrain_D
            self.pretrain = self.pretrain_D > 0
            self.loss_G_fake = 0
            self.loss_matching = 0
            self.loss_D_budget_loss = 0
            self.loss_G_CE = 0
            self.norms_G = []
            self.iteration_G = 0
            self.iteration_D = 0

            self.tile_size = 128
            if self.opt.zero_prediction:
                self.visual_names.append("bet_map2")
                self.loss_names.append("D_bet_loss_zero")

            if not os.path.exists("checkpoints/" + opt.name + "/gradients"):
                os.mkdir("checkpoints/" + opt.name + "/gradients") 

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.borders = input['C'].to(self.device).float()
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A) # Forward pass        

    def forward_D(self, input_D, zero_pred=False, zero_rgb=False, channel=False, gen=False):
        self.mask = (self.real_B != self.ignore).unsqueeze(dim=1).expand(self.fake_B.shape).float() # mask for ignore class  
        input_D = (input_D * self.mask) - (1 - self.mask)

        if zero_rgb:
            fake_AB = torch.cat((torch.zeros(self.real_A.shape).to(self.device), input_D), 1) 
        else:
            fake_AB = torch.cat((self.real_A, input_D), 1)  # concat input rgb + masked output

        output_gambler = self.netD(fake_AB)
        match = self.sigmoid(output_gambler[:, 1]) # matching channel
        output_gambler = output_gambler[:, 0] ## Bet map

        if zero_pred:
            self.errors = (torch.argmax(self.fake_B, dim=1) != self.real_B).float() * (self.real_B != self.ignore).float()
            loss_bet = torch.mean(self.criterionBCE(output_gambler.squeeze(dim=1), self.errors) * (1-self.borders))
        elif gen:
            self.prediction = input_D
            self.errors = (torch.argmax(input_D, dim=1) != self.real_B).float() * (self.real_B != self.ignore).float()
            loss_bet = -torch.mean(self.criterionBCE(output_gambler.squeeze(dim=1), self.errors))
        else:
            self.prediction = input_D
            self.errors = (torch.argmax(input_D, dim=1) != self.real_B).float() * (self.real_B != self.ignore).float()
            loss_bet = torch.mean(self.criterionBCE(output_gambler.squeeze(dim=1), self.errors) * (1-self.borders))  #self.criterionMSE(output_gambler, errors)
            if channel and self.opt.matching:
                size = match.shape
                label = torch.ones(match.shape).to(self.device) * self.match_label
                loss_matching = self.criterionmatch(match, label)
                prediction = torch.mean(match.reshape(size[0], -1), dim=1) > 0.5
                target = (torch.ones(size[0]).byte() * self.match_label).to(self.device)
                self.correct.append((prediction == target).sum().item())
                self.total.append(size[0])
                self.bet_map2 = match.clone().detach()
                if len(self.correct) > 100:
                    self.accuracy_matching.append(sum(self.correct)/sum(self.total))
                    del self.correct[0]
                    del self.total[0]
                return output_gambler, loss_bet, loss_matching
        return output_gambler, loss_bet

    def to_one_hot(self, labels, C):
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        target = one_hot.scatter_(1, labels.cpu(), 1)
        return target.cuda().float()
            
    def hist_gradient(self, parameters, name_model):
        counter = 0
        store = []
        for name , p in parameters:
            if len(p.shape) > 3:
                store.append(("conv" + str(counter) + "_" + str(p.shape), p.flatten().cpu().data))
                counter += 1
        fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(15,15), constrained_layout=True)
        counter = 0
        if name == "G":
            img_path = "checkpoints/" + self.opt.name + "/gradients/histogram_iter" + str(self.iteration_G) + "_" + name_model + ".png"
        else:
            img_path = "checkpoints/" + self.opt.name + "/gradients/histogram_iter" + str(self.iteration_D) + "_" + name_model + ".png"
        for row in ax:
            for col in row:
                if counter < len(store):
                    temp = col.hist(store[counter][1], bins=1000)
                    col.set_title(store[counter][0])
                    counter += 1
        plt.savefig(img_path, dpi=100)
        plt.close()

    def shuffle_input(self):
        size = self.fake_B.shape
        if np.random.uniform() > 0.5:
            self.match_label = 0 # Not matching
            self.real_A = self.real_A[np.arange(-1, size[0] - 1)]
        else:
            self.match_label = 1 # Are matching

    def normalize_data(self, inp):
        out = (inp - - 1)/((self.n_classes - 1) - -1)
        # out = (inp - torch.mean(inp))/torch.std(inp)
        return out

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # Train with fake or add ground-truth
        # self.fake_B_masked = self.to_one_hot(torch.argmax(self.fake_B.clone().detach(), dim=1).unsqueeze(dim=1), self.n_classes)
        self.fake_B_masked = torch.argmax(self.fake_B.clone().detach(), dim=1)

        # self.fake_B_masked = ((1 - self.fake_B_masked) * -self.softmax(self.fake_B.detach())) + (self.fake_B_masked - (self.fake_B_masked * self.softmax(self.fake_B.detach())))
        # self.fake_B_masked = self.softmax(self.fake_B) + self.fake_B_masked
        # class_ids = torch.arange(19).reshape(19, 1, 1,).expand(19, 512, 512).to(self.device)
        # batch_class_ids = torch.stack([class_ids for i in range(self.fake_B_masked.shape[0])]).float()
        # self.fake_B_masked = torch.sum(self.fake_B_masked * batch_class_ids, dim=1)
        mask = (self.real_B != self.ignore).float() # mask for ignore class 
        self.real_B_masked = (self.real_B.float() * mask) - (1 - mask) 
        self.fake_B_masked = (self.fake_B_masked.float() * mask) - (1 - mask) 
        self.fake_B_norm = self.normalize_data(self.fake_B_masked)
        self.real_B_norm = self.normalize_data(self.real_B_masked.float())


        self.optimizer_D.zero_grad()
        fake_AB = torch.cat((self.real_A, self.fake_B_norm.unsqueeze(dim=1)), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB).squeeze()
        self.loss_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B_norm.unsqueeze(dim=1)), 1)
        pred_real = self.netD(real_AB).squeeze()

        self.loss_real = self.criterionGAN(pred_real, True)
         
        # combine loss and calculate gradients
        self.loss_D = (self.loss_fake + self.loss_real) * 0.5
        self.loss_D.backward()
        self.optimizer_D.step()
        # self.ce_map = self.errors

        if self.iteration_D % 100 == 0:
            self.hist_gradient(list(self.netD.named_parameters()), "D")

        self.iteration_D += 1
        self.bet_map = self.sigmoid(pred_fake).squeeze(dim=1)
        self.bet_map2 = self.sigmoid(pred_real).squeeze(dim=1)

    
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.fake_B_masked = self.to_one_hot(torch.argmax(self.fake_B.clone().detach(), dim=1).unsqueeze(dim=1), self.n_classes)
        self.fake_B_masked = ((1 - self.fake_B_masked) * -self.softmax(self.fake_B.detach())) + (self.fake_B_masked - (self.fake_B_masked * self.softmax(self.fake_B.detach())))
        self.fake_B_masked = self.softmax(self.fake_B) + self.fake_B_masked
        class_ids = torch.arange(19).reshape(19, 1, 1,).expand(19, 512, 512).to(self.device)
        batch_class_ids = torch.stack([class_ids for i in range(self.fake_B_masked.shape[0])]).float()
        self.fake_B_masked = torch.sum(self.fake_B_masked * batch_class_ids, dim=1)
        mask = (self.real_B != self.ignore).float() # mask for ignore class 
        self.real_B_masked = (self.real_B.float() * mask) - (1 - mask) 
        self.fake_B_masked = (self.fake_B_masked.float() * mask) - (1 - mask) 
        self.fake_B_norm = self.normalize_data(self.fake_B_masked)
        self.real_B_norm = self.normalize_data(self.real_B_masked.float())


        self.optimizer_D.zero_grad()
        fake_AB = torch.cat((self.real_A, self.fake_B_norm.unsqueeze(dim=1)), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB).squeeze()
        self.loss_G_fake = self.criterionGAN(pred_fake, True)
        self.loss_G_CE = self.criterionCE(self.fake_B, self.real_B) # Backpropgate through Cross Entropy here
        self.loss_G = self.loss_G_fake + self.loss_G_CE
        self.loss_G.backward()
        total_norm = 0
        for p in self.netG.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.norms_G.append(float(total_norm))
        if self.iteration_G % 25 == 0:
            self.hist_gradient(list(self.netG.named_parameters()), "G")
        self.iteration_G += 1
        
    def optimize_parameters(self):
        # compute fake images: G(A)
        self.forward() 
        if self.pretrain == True:
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.backward_D()                # calculate gradients for D
            self.count_D += 1
            who_trains = "pre_D"
            if self.pretrain_D == self.count_D:
                self.pretrain = False
                self.count_D = 0
                print("finished the pretraining")
        else:
            if self.count_G < self.G_train:
                self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
                self.optimizer_G.zero_grad()        # set G's gradients to zero
                self.backward_G()                   # calculate graidents for G
                self.optimizer_G.step()             # udpate G's weights
                self.count_G += 1
                who_trains = "G" + str(self.count_G) + "-" + str(self.G_train)
                if self.count_G == self.G_train:
                    print("finished scheduling G: Evaluation mode")
            else:
                self.set_requires_grad(self.netD, True)  # enable backprop for D
                self.backward_D()                # calculate gradients for D
                self.count_D += 1                # update D
                who_trains = "D" + str(self.count_D) + "-" + str(self.D_train)
                if self.count_D == self.D_train:
                    self.count_G = 0
                    self.count_D = 0
                    print("finished scheduling D: Training mode")

        # Creating visual output
        self.fake_B_output = torch.argmax(self.fake_B[0].unsqueeze(dim=0), dim=1)
        self.fake_B_output[(self.real_B[0] == self.ignore).unsqueeze(dim=0)] = self.real_B[0][self.real_B[0] == self.ignore].unsqueeze(dim=0)
        # self.prediction = self.fake_B_masked
        # if self.opt.matching:
        #     self.interleaved = torch.argmax(self.fake_B_masked[0].unsqueeze(dim=0), dim=1)
        #     self.interleaved[(self.real_B[0] == self.ignore).unsqueeze(dim=0)] = self.real_B[0][self.real_B[0] == self.ignore].unsqueeze(dim=0)

        if self.count_G > 0 and self.count_G % 100 == 0:
            plt.plot(self.norms_G)
            plt.title("Gradient norm G")
            plt.savefig("checkpoints/" + self.opt.name + "/norm_G.png")
            plt.close()

        if self.count_D > 0 and self.count_D % 25 == 0 and self.opt.matching:
            plt.plot(self.accuracy_matching)
            plt.title("Accuracy matching")
            plt.savefig("checkpoints/" + self.opt.name + "/matching.png")
            plt.close()
        self.netG.train()
        total = 0
        return who_trains