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



class CEModel(BaseModel):
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
            parser.add_argument('--interleave', type=int, default=0, help='train on the ce map')
            parser.add_argument('--thickness', type=int, default=1, help='train on the ce map')
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
        self.visual_names = ['real_A', 'real_B', 'fake_B_output', "borders", 'bet_map', 'ce_map', "prediction"]
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
            output_nc = 1
            opt.init_type = "kaiming"
            opt.init_gain = 0.25
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, output_nc)
            # state_dict_D = torch.load("latest_net_D.pth", map_location=self.device)
            # self.netD.load_state_dict(state_dict_D)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=5e-4)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # Initalize loss functions
            self.weight = torch.tensor([2.5, 24.1, 4.9, 209.9, 151.4, 86.8, 498.4, 186.2, 6.12, 113.6, 17.2, 82.9, 632.9, 16.9, 446.,411.5, 450.2, 1158.9, 274.6]).to(self.device)
            self.criterionCE = torch.nn.CrossEntropyLoss(weight=self.weight**(self.opt.alpha), ignore_index=self.ignore) 
            # self.criterionCE = FocalLoss2d(weight=self.weight**(0.25), gamma = 2)
   
            self.BCEweight = torch.tensor([self.opt.weight_gambler]).to(self.device)
            self.criterionBCE = torch.nn.BCEWithLogitsLoss(pos_weight=self.BCEweight, size_average=True)

            # layers for output U-nets
            self.softmax = torch.nn.Softmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()

            # Loading the generator        
            if opt.pretrained == "full":
                state_dict = torch.load("latest_net_G_full.pth", map_location=self.device)
                self.netG.load_state_dict(state_dict)
            elif opt.pretrained == "split":
                state_dict = torch.load("latest_net_G_split.pth", map_location=self.device)
                self.netG.load_state_dict(state_dict)
            else:
                print("Starting from scratch")

            if self.opt.zero_prediction:
                self.loss_names.extend(["D_bet_loss_zero"])
                self.visual_names.append("bet_map2")

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

            self.tile_size = 32

            if not os.path.exists("checkpoints/" + opt.name + "/gradients"):
                os.mkdir("checkpoints/" + opt.name + "/gradients")


            model_urls = {
                    'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
                    'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
                    'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
            }

            shape_model = torchvision.models.resnet50(pretrained=False)
            
            self.shape_model = torch.nn.DataParallel(shape_model).cuda()
            checkpoint = model_zoo.load_url(model_urls['resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN'])
            self.shape_model.load_state_dict(checkpoint["state_dict"])
            self.res50_conv = torch.nn.Sequential(*list(self.shape_model.module.children())[:-5])
            self.upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
            self.shape_model = torch.nn.Sequential(*list(self.shape_model.module.children())[:-3])

            self.decoder = []
            channels = [256, 128, 64, 64, 19]
            for i in range(4):
                upconv = torch.nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1, bias=True)
                uprelu = torch.nn.ReLU(True)
                norm = torch.nn.BatchNorm2d(channels[i + 1])
                if i == 3:
                    upsample = [upconv]
                else:
                    upsample = [upconv, norm, uprelu]
                self.decoder += upsample
            self.decoder = torch.nn.Sequential(*list(self.decoder)).to(self.device)
            self.optimizer_res = torch.optim.Adam(self.res50_conv.parameters(), lr=opt.lr_D, betas=(0.9, 0.999))
            self.optimizer_decoder = torch.optim.Adam(self.decoder.parameters(), lr=opt.lr_D, betas=(0.9, 0.999))
                      
    

    def interleave(self):
        one_hot_real = self.real_B.clone()
        one_hot_real[one_hot_real == self.ignore] = 0
        one_hot_real = self.to_one_hot(one_hot_real.unsqueeze(dim=1), 19).float()
        size = self.real_B.shape
        tile_mask = torch.zeros(size).to(self.device)
        for i in range(0, 512, self.tile_size):
            for j in range(0, 512, self.tile_size):
                tile_mask[:, i:i+self.tile_size, j:j+self.tile_size] = torch.bernoulli(torch.ones(size[0], 1, 1) * 0.5).expand(size[0], self.tile_size, self.tile_size)
        tile_mask = tile_mask.unsqueeze(dim=1).expand(self.fake_B.shape)
        self.fake_B_masked[tile_mask.byte()] = one_hot_real[tile_mask.byte()]

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
        self.netG.eval()
        self.fake_B = self.netG(self.real_A)  # G(A) # Forward pass
        self.mask = (self.real_B == self.ignore).unsqueeze(dim=1).expand(self.fake_B.shape) # mask for ignore class, zeroes out the ignore class   
        # max_indices = self.to_one_hot(torch.argmax(self.fake_B.clone().detach(), dim=1).unsqueeze(dim=1), 19) * 100 # N x 512 X 512 --> N X 1 X 512 X 512 --> N X 19 X 512 X 512 --> * 100
        self.fake_B_masked = self.to_one_hot(torch.argmax(self.fake_B.clone().detach(), dim=1).unsqueeze(dim=1), 19)
        if self.opt.interleave:
            self.interleave()
        self.fake_B_masked[self.mask] = -1 

            
    def to_one_hot(self, labels, C):
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        target = one_hot.scatter_(1, labels.cpu(), 1)
        return target.cuda().float()

    def get_borders(self, image):
        size = image.shape
        image = self.to_one_hot(torch.argmax(image, dim=1).unsqueeze(dim=1), 19).float().cpu().numpy()
        borders = torch.zeros(size[0], size[2], size[3]).to(self.device)
        for i in range(size[0]):
            for j in range(size[1]):
                borders[i] += torch.from_numpy(image[i, j] - scipy.ndimage.binary_erosion(image[i, j], iterations=self.opt.thickness).astype(np.int64)).float().to(self.device)
        return borders.float()

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
        if not self.netD.training:
            print("Gambler is in evaluation mode:")
        self.borders = self.get_borders(self.fake_B_masked.clone().detach()) # N X 512 X 512
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        # self.optimizer_res.zero_grad()
        # shape_featurs_A = self.res50_conv(self.real_A)


        # self.mask_down = self.fake_B_masked.clone()
        # self.mask_down[self.mask_down < 0] = 0
        # for i in range(self.mask_down.shape[1]):
        #     self.mask_down[:, i] = self.mask_down[:, i] * (i + 1)
        # # self.fake_B_down = self.mask_down.sum(dim=1)
        # # self.fake_B_down[self.real_B == self.ignore] = self.n_classes + 1

        # self.fake_B_masked[self.fake_B_masked < 0] = 0
        # self.fake_B_masked = self.fake_B_masked + self.mask_down.detach()
        # self.fake_B_masked = self.fake_B_masked.sum(dim=1)
        # self.fake_B_masked[self.real_B == self.ignore] = self.n_classes + 1
        # self.fake_B_masked = ((self.fake_B_masked - torch.min(self.fake_B_masked))/(torch.max(self.fake_B_masked) - torch.min(self.fake_B_masked))).unsqueeze(dim=1)

        fake_AB = torch.cat((self.real_A, self.fake_B_masked), 1)  # concat input rgb + masked output
        output_gambler = self.netD(fake_AB.detach()).squeeze()

        errors = (torch.argmax(self.fake_B, dim=1) != self.real_B).float() * (self.real_B != self.ignore).float() * (1 - self.borders)
        self.loss_D_bet_loss = self.criterionBCE(output_gambler, errors)  #self.criterionMSE(output_gambler, errors)
        self.loss_D_bet_loss.backward()
        self.optimizer_D.step() 
        # self.optimizer_res.step()

        output_gambler = self.sigmoid(output_gambler.detach())
        
        # if self.opt.zero_prediction:
        #     fake_AB_zero = torch.cat((self.real_A, -10 * torch.ones(self.fake_B_masked.shape).to(self.device)), 1)
        #     output_gambler_zero = self.netD(fake_AB_zero).squeeze()
        #     self.optimizer_D.zero_grad() 
        #     self.loss_D_bet_loss_zero = self.criterionBCE(output_gambler_zero, errors)
        #     self.loss_D_bet_loss_zero.backward()
        #     self.optimizer_D.step()   
        #     output_gambler_zero = self.sigmoid(output_gambler_zero)
        #     self.bet_map2 = output_gambler_zero
                      
        if self.iteration_D % 100 == 0:
            self.hist_gradient(list(self.netD.named_parameters()), "D")
            # for i in range(256):
            #     plt.imshow(shape_featurs_A[0, i].detach().cpu().numpy(), cmap = 'gray', vmin=0, vmax=1)
            #     plt.savefig("features/shape_features/" + str(i) + ".png")
            #     plt.close()
        self.iteration_D += 1
        self.bet_map = output_gambler
        self.ce_map = errors

    
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        self.fake_B_masked = self.softmax(self.fake_B.clone()) * (1 - self.mask.float()) - self.mask.float()
        fake_AB = torch.cat((self.real_A, self.fake_B_masked), 1)
        self.borders = self.get_borders(self.fake_B_masked.clone().detach()) # N X 512 X 512

        # errors = self.criterionBET(self.fake_B, self.real_B) * (self.real_B != self.ignore).float()
        output_gambler = self.netD(fake_AB).squeeze() # N X 1 X 512 X 512, training discr
        errors = (torch.argmax(self.fake_B.detach(), dim=1) != self.real_B).float() * (self.real_B != self.ignore).float()
        self.loss_G_bet_loss = -self.criterionBCE(output_gambler, errors)  #self.criterionMSE(output_gambler, errors)
        self.loss_G_CE = self.criterionCE(self.fake_B, self.real_B) # Backpropgate through Cross Entropy here
        self.loss_G = self.loss_G_bet_loss + self.loss_G_CE
        self.loss_G.backward()
        self.bet_map = self.sigmoid(output_gambler)
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
                self.backward_D()                # calculate gradients for D
                # self.optimizer_D.step()          # update D's weights
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
        self.fake_B_output = torch.argmax(self.fake_B[0].unsqueeze(dim=0), dim=1)
        self.fake_B_output[(self.real_B[0] == self.ignore).unsqueeze(dim=0)] = self.real_B[0][self.real_B[0] == self.ignore].unsqueeze(dim=0)
        self.prediction = self.fake_B_masked

        plt.plot(self.norms_G)
        plt.title("Gradient norm G")
        plt.savefig("checkpoints/" + self.opt.name + "/norm_G.png")
        plt.close()
        self.netG.train()
        return who_trains