import torch
from .base_model import BaseModel
from . import networks


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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_EL', type=float, default=1.0, help='weight for Embedding loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['EL', 'fit', 'D_fake', 'D_real']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.opt = opt
        if self.isTrain:
            # Number of pretrain iterations
            self.pretrain_G = opt.pretrain_G
            self.count_G = 0

            self.pretrain_D = opt.pretrain_D
            self.count_D = 0
            self.pretrain = self.pretrain_D > 0 or self.pretrain_D > 0

            # Initalize the loss functions for plot
            self.loss_D_real = 0
            self.loss_D_fake = 0
            self.loss_EL = 0
            self.loss_fit = 0
            # Number of steps the generator and discriminator train in a row 
            self.G_train = opt.G_train
            self.D_train = opt.D_train

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionEL = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # Statistics of discriminator
            self.correct_predictions = 0

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
        self.fake_B = self.netG(self.real_A)  # G(A)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # pred_fake = self.netD.forward(fake_AB.detach())
        # self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # # Real
        # real_AB = torch.cat((self.real_A, self.real_B), 1)
        # pred_real = self.netD.forward(real_AB)
        # self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        
        # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        fake_AB = torch.cat((self.real_A, self.fake_B), 1).detach()
        emb_fake = self.netD.forward(fake_AB, emb=True)
       
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        emb_real = self.netD.forward(real_AB, emb=True)
        self.loss_EL = - self.criterionEL(emb_fake, emb_real) 
        self.loss_D = self.loss_EL

        self.loss_D.backward()

    def backward_G(self):
        """Calculate embedding loss and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        if self.pretrain == False:
            # Embedding loss
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            emb_fake = self.netD.forward(fake_AB, emb=True)
           
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            emb_real = self.netD.forward(real_AB, emb=True).detach()
            self.loss_EL = self.criterionEL(emb_fake, emb_real.detach()) * self.opt.lambda_EL/(self.opt.lambda_EL + 1) 
            # self.loss_EL = 0
            # for i in range(emb_fake.shape[1]):
            #     self.loss_EL += self.criterionEL(emb_fake[:, i], emb_real[:, i]) * self.opt.lambda_L1/(self.opt.lambda_L1 + 1) ## Pix2PixHD

        # Second, G(A) = B

       
        # combine loss and calculate gradients
        if self.pretrain:
            self.loss_fit = self.criterionL1(self.fake_B, self.real_B) 
            self.loss_G = self.loss_fit
        else:
            self.loss_fit = 1/(self.opt.lambda_EL + 1) * self.criterionL1(self.fake_B, self.real_B) 
            self.loss_G = self.loss_EL + self.loss_fit
        self.loss_G.backward()

    def optimize_G(self):
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def optimize_D(self):
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        if self.pretrain:
            if self.count_G < self.pretrain_G:
                self.optimize_G()
                self.count_G += 1 * self.opt.batch_size
            elif self.count_D < self.pretrain_D:
                self.optimize_D()
                self.count_D += 1 * self.opt.batch_size
            else:
                print("Finished the pretraining")
                self.pretrain = False
                self.count_D = 0
                self.count_G = 0
        else:
            if self.count_G < self.G_train:
                self.optimize_G()
                self.count_G += 1 
            elif self.count_D < self.D_train:
                self.optimize_D()
                self.count_D += 1
            else:
                self.count_D = 0
                self.count_G = 1
                self.optimize_G()



