import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from scripts.eval_cityscapes import *
from inception_score import *
import numpy as np
from sklearn.metrics import confusion_matrix  
import pickle
from torch.utils.data import DataLoader
from models import networks
import cityscapes
from cityscapes import evaluate, colorize_mask
import torchvision
import cv2
import scipy.misc
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import voc
from PIL import Image
import torchvision
import util.transforms as extended_transforms


def save_images(images, image_path):
    x_offset = 0
    widths, heights = zip(*[i.size for i in images])

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    for image in images:
        new_im.paste(image, (x_offset,0))
        x_offset += image.size[0]
    new_im.save(image_path)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return np.array(input_image)
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.ndim == 2:
            image_numpy = cityscapes.visualize(colorize_mask(image_numpy).convert("RGB")).numpy()
        elif image_numpy.ndim == 3:
            image_numpy = (image_numpy - np.min(image_numpy))/(np.max(image_numpy)-np.min(image_numpy))
        elif image_numpy.ndim == 4:
            image_numpy = (image_numpy - np.min(image_numpy))/(np.max(image_numpy)-np.min(image_numpy))
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = np.array(ninput_image)
    return image_numpy.astype(imtype)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # val_set = cityscapes.CityScapes('fine', opt.phase, joint_transform=cityscapes.val_joint_transform, transform=cityscapes.input_transform,
    #                                 target_transform=cityscapes.target_transform)
    # dataset = DataLoader(val_set, batch_size=1, num_workers=8, shuffle=False)
    if opt.dataset == "voc":
        train_set = voc.VOCAug(split='val', augment=False, scales=(1.0, 1.0), flip=False, crop_size=513)
        # val_set = voc.VOC('val', joint_transform=voc.val_joint_transform, transform=voc.input_transform,
        #                             target_transform=voc.target_transform)
        # val_loader = DataLoader(val_set, batch_size=opt.batch_size, num_workers=8, shuffle=False)
        opt.output_nc = 21
        cityscapes.num_classes = 21
        cityscapes.classes = voc.classes
        cityscapes.palette = voc.palette
    else:
        train_set = cityscapes.CityScapes('fine', opt.phase, joint_transform=cityscapes.val_joint_transform,
                                          transform=cityscapes.input_transform, target_transform=cityscapes.target_transform)
    
    dataset = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=False)
    model = create_model(opt)      # create a model given opt.model and other options
    print(model)
    # model.setup(opt)               # regular setup: load and print networks; create schedulers

    # Unet
    seg = "cityscapes_segementation_no_flip_lrG00001_lrD00002_D1_L3_G-D50-100_ndf64_lamb10_GT00"
    # v2 = "exploit4_scratch_lrG00001_lrD00001_D1_L3_G-D200-400_ndf64_lamb025_GT00"
    v1 = "u_net_scratch_lrG00001_lrD00001_D1_L3_G-D200-400_ndf64_lamb10_GT00"
    # el ="el_scratch_lrG00001_lrD00001_D1_L3_G-D200-100_ndf64_lamb001_GT00"
    # pix = "cs_p2p_scratch_lrG00001_lrD00005_D1_L3_G-D200-100_ndf64_lamb001_GT00"
    filename = "checkpoints/" + opt.dataset + "/" + "betting" + "/" + v1 + "/latest_net_D.pth"
    state_dict = torch.load(filename, map_location=model.device)
    model.netD.load_state_dict(state_dict)
    restore_transform = torchvision.transforms.Compose([
        extended_transforms.DeNormalize(*cityscapes.mean_std),
        ])
    # Psp
    # seg = "psp025_scratch_lrG5e-05_lrD00005_D1_L3_G-D50-100_ndf64_lamb10_GT00"
    # v2 = "psp_early_lrG5e-06_lrD5e-06_D1_L3_G-D800-800_ndf64_lamb10_GT00"
    # v1 = "psp_scratch_lrG25e-05_lrD25e-05_D1_L3_G-D800-800_ndf64_lamb10_GT00"
    # el = "psp_cs_early_lrG1e-06_lrD00001_D1_L3_G-D800-400_ndf128_lamb001_GT00"
    # pix = "psp_cs_early_lrG1e-06_lrD00001_D1_L3_G-D800-400_ndf128_lamb001_GT00"
    best_models = [("seg", seg), ("betting", v1)]
    images = []
    for model_inf in best_models:
        filename = "checkpoints/" + opt.dataset + "/" + model_inf[0] + "/" + model_inf[1] + "/latest_net_G.pth"
        state_dict = torch.load(filename, map_location=model.device)
        model.netG.load_state_dict(state_dict)
            


        # create a website
        model.ignore = 255
        # web_dir = os.path.join(opt.results_dir, "evaluate_" + opt.name + opt.suffix2, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
        # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % ("evaluate_" + opt.name, opt.phase, opt.epoch))
      
        # for layer in model.netD.model.modules():
        #     if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.LeakyReLU) or isinstance(layer, torch.nn.ConvTranspose2d) or isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.ReLU):
        #         layers.append(layer)
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

        sigmoid = torch.nn.Sigmoid()
        softmax = torch.nn.Softmax(dim=1)
        # Visualize conv filter
        model.visual_names = ["real_A", "real_B"]
        model.eval()
        data = {}
        print(model_inf[0])
        for i, data2 in enumerate(dataset):
            data["A"] = data2[0].cuda()
            data["B"] = data2[1].cuda()
            # data["C"] = data2[2].cuda()
            data["A_paths"] = '%04d.png' % i
            model.set_input(data)  # unpack data from data loader
            with torch.no_grad():
                model.forward()

            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            prediction = torch.argmax(model.fake_B, dim=1)
            prediction[(model.real_B == 255)] = 255
            if model_inf[0] == "seg":
                mask = (model.real_B != model.ignore).unsqueeze(dim=1).expand(model.fake_B.shape).float() # mask for ignore class, zeroes out the ignore class    
                fake_B_masked = model.softmax(model.fake_B.clone()) * mask - (1 - mask)
                fake_AB = torch.cat((model.real_A, fake_B_masked), 1) 
                betting_map = model.netD(fake_AB).squeeze(dim=1)
                image_numpy = sigmoid(betting_map.data[0]).cpu().numpy()
                sizes = image_numpy.shape
                fig = plt.figure()
                fig.set_size_inches(1. * sizes[1]/sizes[0], 1, forward=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(image_numpy, cmap = 'gray', vmin=0, vmax=1)
                plt.savefig("temp.png", dpi=sizes[0])
                plt.close()
                betting_map = Image.open("temp.png")
                os.remove("temp.png")
                rgb = tensor2im(restore_transform(data2[0].cpu()))
                GT = tensor2im(data2[1].cpu())
                rgb = Image.fromarray(rgb).convert("RGB")
                GT = Image.fromarray(GT).convert("RGB")
                filename_rgb = "Final_figures/first_image/" + str(i) + "_rgb.png"
                filename_GT = "Final_figures/first_image/" + str(i) + "_GT.png"
                filename_betmap = "Final_figures/first_image/" + str(i) + "_betmap.png"
                rgb.save(filename_rgb)
                GT.save(filename_GT)
                betting_map.save(filename_betmap)

            image_numpy = tensor2im(prediction)
            image_pil = Image.fromarray(image_numpy).convert("RGB")
            image_path = "Final_figures/first_image/" + str(i) + "_" + model_inf[0] + ".png"
            image_pil.save(image_path)

