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
import camvid
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
    if opt.dataset == "camvid":
        train_set = camvid.Camvid('val', joint_transform=camvid.val_joint_transform,
                                          transform=camvid.input_transform, target_transform=camvid.target_transform)
        opt.output_nc = 11
        cityscapes.num_classes = 11
        cityscapes.classes = camvid.classes
        cityscapes.palette = camvid.palette
        cityscapes.mean_std = camvid.mean_std
    else:
        train_set = cityscapes.CityScapes('fine', opt.phase, joint_transform=cityscapes.val_joint_transform,
                                          transform=cityscapes.input_transform, target_transform=cityscapes.target_transform)
    
    dataset = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=False)
    model = create_model(opt)      # create a model given opt.model and other options
    print(model)
    # model.setup(opt)               # regular setup: load and print networks; create schedulers

    # Unet
    # seg = "cityscapes_segementation_no_flip_lrG00001_lrD00002_D1_L3_G-D50-100_ndf64_lamb10_GT00"
    # v2 = "exploit4_scratch_lrG00001_lrD00001_D1_L3_G-D200-400_ndf64_lamb025_GT00"
    # v1 = "u_net_scratch_lrG00001_lrD00001_D1_L3_G-D200-400_ndf64_lamb10_GT00"
    # el ="el_scratch_lrG00001_lrD00001_D1_L3_G-D200-100_ndf64_lamb001_GT00"
    # pix = "cs_p2p_scratch_lrG00001_lrD00005_D1_L3_G-D200-100_ndf64_lamb001_GT00"

    # Psp
    # restore_transform = torchvision.transforms.Compose([
    #     extended_transforms.DeNormalize(*cityscapes.mean_std),
    #     ])
    ## UNET CITYSCAPES
    seg = "cityscapes_segementation_no_flip_lrG00001_lrD00002_D1_L3_G-D50-100_ndf64_lamb10_GT00"
    focal = "cityscapes_segementation_flip_focal_lrG00001_lrD00002_D1_L3_G-D50-100_ndf64_lamb10_GT00"
    pix = "cs_p2p_scratch_lrG00001_lrD00005_D1_L3_G-D200-100_ndf64_lamb001_GT00"
    el = "el_scratch_lrG00001_lrD00001_D1_L3_G-D200-100_ndf64_lamb001_GT00"
    HD = "unet_cs_HD_scratch_lrG00001_lrD00005_D2_L3_G-D200-100_ndf128_lamb001_GT00"
    v1 = "u_net_scratch_lrG00001_lrD00001_D1_L3_G-D200-400_ndf64_lamb10_GT00"
    v2 = "exploit4_scratch_lrG00001_lrD00001_D1_L3_G-D200-400_ndf64_lamb025_GT00"

    # ## PSP-NET CITYSCAPES
    # seg = ""
    # focal = ""
    # v2 = ""
    # v1 = ""
    # el = ""
    # HD = ""
    # pix = ""

    # ## PSP_NET CAMVID
    # seg = ""
    # focal = ""
    # v2 = ""
    # v1 = ""
    # el = ""
    # HD = ""
    # pix = ""
    # THesis
    best_models = [("seg", seg), ("seg", focal), ("pix2pix", pix), ("el", el), ("el", HD), ("betting", v1), ("ce", v2)]
    names = ["Focal loss", "CE + adv", "EL-GAN", "MD EL-GAN", "Gambling nets", "Gambling nets - arg"]

    # # paper
    # best_models = [("seg", seg), ("seg", focal), ("pix2pix", pix), ("el", el), ("betting", v1)]
    # names = ["Focal loss", "CE + adv", "EL-GAN", "Gambling nets"]


    images = []
    results = []
    for model_inf in best_models:
        print(model_inf)
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
        predictions = []
        labels = []
        data = {}
        model_inf = list(model_inf)
        if "HD" in model_inf[1]:
            model_inf[0] = "HD"
        if "focal" in model_inf[1]:
            model_inf[0] = "focal"
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
            predictions.append(prediction.squeeze().cpu())
            labels.append(model.real_B.squeeze().cpu())
        acc, acc_cls, mean_iu, iu, _ = evaluate(torch.stack(predictions).numpy(), torch.stack(labels).numpy(), cityscapes.num_classes)
        results.append((iu, model_inf[0]))
        print(mean_iu)


    n_groups = len(iu)
    baseline_iu, name = results[0]
    del results[0]
    fig = plt.figure(figsize=(20, 10))
    index = np.arange(n_groups)
    bar_width = 1/(len(results) + 1)
    classes = list(reversed(['road', 'vegetation', 'building', 'sky', 'car', 'train', 'bus', 'sidewalk', 'truck', 'fence', 'wall', 'person', 'trafficsign', 'terrain', 'bicycle', 'motorcycle', 'rider', 'trafficlight', 'pole']))
    for i, (iu, name) in enumerate(results):    
        plt.bar(index, iu - baseline_iu, bar_width, label=names[i], alpha=0.8)
        index = index + bar_width
    plt.ylabel('delta IoU per class')
    plt.xlabel('class names')
    index = np.arange(n_groups)

    plt.xticks(index + ((0.5 * len(results)) * bar_width), classes, rotation='vertical')

    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig("test_bar_plot.png", dpi=100)
    
            
            

