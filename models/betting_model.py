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




class BettingModel(BaseModel):
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
            # parser.add_argument('--lambda_budget', type=float, default=0.00003, help='weight for L1 loss')
            parser.add_argument('--budget', type=float, default=3000, help='budget for discriminator')
            parser.add_argument('--alpha', type=float, default=0.25, help='Coefficient for weighting')
            parser.add_argument('--freeze', type=int, default=0, help='run with frozen generator for evaluation')
            parser.add_argument('--soft', type=int, default=1, help='use softmax output')
            parser.add_argument('--load_model', type=int, default=1, help='start with pretrained model')
            parser.add_argument('--smooth', type=float, default=0.02, help='smoothing for the betting map')
            parser.add_argument('--argmax', type=int, default=0, help='add argmax to input of gambler')
            parser.add_argument('--input', type=int, default=0, help='only input the rgb as input')
            parser.add_argument('--peaked', type=int, default=0, help='only input the rgb as input')
            parser.add_argument('--zero_prediction', type=int, default=0, help='train on both the rgb only and rgb + prediction')
            parser.add_argument('--ce_loss', type=int, default=0, help='train on the ce map')




        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_bet_loss', 'G_CE', 'D_bet_loss', 'D_budget_loss']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_B_output', 'bet_map', 'ce_map', "prediction"]
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

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            
            if self.opt.argmax:
                self.netD = networks.define_D(opt.input_nc + (opt.output_nc * 2), opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            elif self.opt.input:
                self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                output_nc = 1
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, output_nc)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=5e-4)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=1e-4)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # Initalize loss functions
            self.weight = torch.tensor([2.5, 24.1, 4.9, 209.9, 151.4, 86.8, 498.4, 186.2, 6.12, 113.6, 17.2, 82.9, 632.9, 16.9, 446.,411.5, 450.2, 1158.9, 274.6]).to(self.device)
            self.criterionBET = torch.nn.CrossEntropyLoss(ignore_index=self.ignore, reduction="none")
            self.criterionBET_weight = torch.nn.CrossEntropyLoss(weight=self.weight**(0.25), ignore_index=self.ignore, reduction="none")

            self.criterionCE = torch.nn.CrossEntropyLoss(weight=self.weight**(self.opt.alpha), ignore_index=self.ignore) 
            # self.criterionCE = FocalLoss2d(weight=self.weight**(0.25), gamma = 2)
   
            self.criterionMSE = torch.nn.MSELoss()

            # layers for output U-nets
            self.softmax = torch.nn.Softmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()
            self.relu = torch.nn.ReLU()

            # Loading the generator        
            self.budget = np.linspace(self.opt.budget, 0, 100)
            if opt.pretrained == "full":
                state_dict = torch.load("latest_net_G_full.pth", map_location=self.device)
                self.netG.load_state_dict(state_dict)
            elif opt.pretrained == "split":
                state_dict = torch.load("latest_net_G_split.pth", map_location=self.device)
                self.netG.load_state_dict(state_dict)
            else:
                print("Starting from scratch")
            if self.opt.freeze:
                self.netG_frozen = copy.deepcopy(self.netG)
                self.visual_names.append("frozen_im")
                self.netG_frozen.eval()

            if self.opt.zero_prediction:
                self.loss_names.extend(["D_bet_loss_zero"])
            self.toPil = torchvision.transforms.ToPILImage()

            # stats
            self.epoch = 0
            self.count_D = 0
            self.count_G = 0
            self.D_train = opt.D_train
            self.G_train = opt.G_train
            self.pretrain_D = opt.pretrain_D
            self.pretrain = self.pretrain_D > 0
            self.loss_G_bet_loss = 0
            self.loss_D_budget_loss = 0
            self.loss_G_CE = 0
            self.norms_G = []
            self.iteration_G = 0
            self.iteration_D = 0
            self.factor = 2
            self.epsilon = 1
            self.count_epsilon = 0
            self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

            if not os.path.exists("checkpoints/" + opt.name + "/gradients"):
                os.mkdir("checkpoints/" + opt.name + "/gradients")


    def annealing_prediction(self, prediction):
        self.count_epsilon += 1
        if self.count_epsilon % 150 == 0 and self.epsilon > 0.1:
            self.epsilon = self.epsilon - 0.05
        if np.random.uniform() > self.epsilon:
            return prediction
        else:
            return torch.zeros(prediction.shape).to(self.device)

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
        self.fake_B = self.netG(self.real_A)  # G(A) # Forward pass
        self.mask = (self.real_B == self.ignore).unsqueeze(dim=1).expand(self.fake_B.shape).float() * -1 # mask for ignore class, zeroes out the ignore class    
        max_indices = self.to_one_hot(torch.argmax(self.fake_B.detach(), dim=1).unsqueeze(dim=1), 19) * 100 # N x 512 X 512 --> N X 1 X 512 X 512 --> N X 19 X 512 X 512 --> * 100
        self.fake_B_masked = self.softmax(self.fake_B.clone() + max_indices) * self.mask # fake * masking

            
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

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # Train with fake or add ground-truth
        if self.opt.input:
            fake_AB = self.real_A
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B_masked), 1)  # concat input rgb + masked output
        
        if self.opt.argmax:
            one_hot_real = self.real_B.clone()
            one_hot_real[one_hot_real == self.ignore] = 0
            one_hot_real = self.to_one_hot(one_hot_real.unsqueeze(dim=1), 19).float()
            fake_AB = torch.cat((fake_AB, one_hot_real), dim=1)

        # fake_AB = torch.cat((fake_AB, torch.argmax(self.fake_B.detach(), dim=1).unsqueeze(dim=1).float()), dim=1)
        errors = self.criterionBET(self.fake_B_masked.detach(), self.real_B) * (self.real_B != self.ignore).float()
        bet_map = self.sigmoid(self.netD(fake_AB.detach())).squeeze() + self.opt.smooth# N X 1 X 512 X 512, training discr
        bet_map = bet_map * (1/(torch.sum(bet_map.reshape(bet_map.shape[0], -1), dim=1)).reshape(bet_map.shape[0], 1, 1).expand(bet_map.shape)) * self.opt.budget
        self.loss_D_bet_loss = -torch.mean(bet_map  * errors)
        self.loss_D = self.loss_D_bet_loss
        if self.opt.zero_prediction:
            fake_AB_zero = torch.cat((self.real_A, torch.zeros(self.fake_B_masked.shape).to(self.device)), 1)
            bet_map_zero = self.sigmoid(self.netD(fake_AB_zero.detach())).squeeze() + self.opt.smooth# N X 1 X 512 X 512, training discr
            bet_map_zero = bet_map_zero * (1/(torch.sum(bet_map_zero.reshape(bet_map_zero.shape[0], -1), dim=1)).reshape(bet_map_zero.shape[0], 1, 1).expand(bet_map_zero.shape)) * self.opt.budget
            self.loss_D_bet_loss_zero = -torch.mean(bet_map_zero * errors)
            self.loss_D += self.loss_D_bet_loss_zero
        self.loss_D.backward()
        self.iteration_D += 1
        if self.iteration_D % 100 == 0:
            self.hist_gradient(list(self.netD.named_parameters()), "D")
        self.bet_map = bet_map
        self.ce_map = errors

    
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.input:
            fake_AB = self.real_A
        else:
            if self.opt.peaked == 0:
                self.fake_B_masked = self.softmax(self.fake_B.clone()) * self.mask # fake * masking
            fake_AB = torch.cat((self.real_A, self.fake_B_masked), 1)

        if self.opt.argmax:
            fake_AB = torch.cat((fake_AB, torch.argmax(self.fake_B.detach(), dim=1).unsqueeze(dim=1).float()), dim=1)

        errors = self.criterionBET(self.fake_B, self.real_B) * (self.real_B != self.ignore).float()
        bet_map = self.sigmoid(self.netD(fake_AB)).squeeze() + self.opt.smooth# N X 1 X 512 X 512, training discr
        bet_map = bet_map * (1/(torch.sum(bet_map.reshape(bet_map.shape[0], -1), dim=1)).reshape(bet_map.shape[0], 1, 1).expand(bet_map.shape)) * self.opt.budget

        self.loss_G_bet_loss = self.opt.lambda_GAN * torch.mean(bet_map * errors)

        self.loss_G_CE = self.criterionCE(self.fake_B, self.real_B) # Backpropgate through Cross Entropy here
        self.loss_G = self.loss_G_bet_loss + self.loss_G_CE

        self.loss_G.backward()
        self.bet_map = bet_map
        total_norm = 0
        for p in self.netG.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.norms_G.append(float(total_norm))
        if self.iteration_G % 25 == 0:
            self.hist_gradient(list(self.netG.named_parameters()), "G")
        self.iteration_G += 1
        self.ce_map = errors

        
    def optimize_parameters(self):
                  # compute fake images: G(A)
        # update D
        self.forward() 
        if self.pretrain == True:
            if self.pretrain_D >= self.count_D: 
                self.set_requires_grad(self.netD, True)  # enable backprop for D
                self.optimizer_D.zero_grad()     # set D's gradients to zero
                self.backward_D()                # calculate gradients for D
                self.optimizer_D.step()          # update D's weights
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
                self.optimizer_D.zero_grad()     # set D's gradients to zero
                self.backward_D()                # calculate gradients for D
                self.optimizer_D.step()          # update D's weights
                self.count_D += 1                # update D
                who_trains = "D" + str(self.count_D) + "-" + str(self.D_train)
                if self.count_D == self.D_train:
                    self.count_G = 0
                    self.count_D = 0
                    print("finished scheduling D: Training mode")

        # Creating visual output
        self.netG.eval()
        if self.opt.freeze:
            with torch.no_grad():
                self.frozen_im = self.netG_frozen(self.real_A[0].unsqueeze(dim=0))
            self.frozen_im = torch.argmax(self.frozen_im, dim=1) 
            self.frozen_im[(self.real_B[0] == self.ignore).unsqueeze(dim=0)] = self.real_B[0][(self.real_B[0] == self.ignore)].unsqueeze(dim=0)
        self.fake_B_output = torch.argmax(self.netG(self.real_A)[0].unsqueeze(dim=0), dim=1)
        self.fake_B_output[(self.real_B[0] == self.ignore).unsqueeze(dim=0)] = self.real_B[0][self.real_B[0] == self.ignore].unsqueeze(dim=0)
        self.prediction = self.fake_B_masked
        # with torch.no_grad():
        #     self.ce_map = self.criterionBET(self.fake_B, self.real_B).detach()
        #     self.ce_map = (self.ce_map - torch.min(self.ce_map))/(torch.max(self.ce_map) - torch.min(self.ce_map))
        plt.plot(self.norms_G)
        plt.title("Gradient norm G")
        plt.savefig("checkpoints/" + self.opt.name + "/norm_G.png")
        plt.close()
        self.netG.train()
        return who_trains