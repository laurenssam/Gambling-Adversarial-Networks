
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from scripts.eval_cityscapes import *
from inception_score import *
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(opt.results_dir, "evaluate_" + opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % ("evaluate_" + opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    dataset_eval = []
    imgs = []
    real_imgs = []
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths


        data_eval = data.copy()
        data_eval["A"] = model.fake_B.clone()
        imgs.append(model.fake_B.clone().cpu().squeeze().numpy())
        real_imgs.append(model.real_B.clone().cpu().squeeze().numpy())
        # print(imgs[0].shape)
        data_eval["original_A"] = visuals["real_B"].clone()

        dataset_eval.append(data_eval.copy())
        # model_eval.set_input(data_eval)
        # model_eval.test()
        # visuals_eval = model_eval.get_current_visuals()
        # img_path_eval = model_eval.get_image_paths()

        # visuals_eval["A_original"] = visuals["real_A"]
        # visuals_eval["data_A"] = visuals["fake_B"]


    if opt.evaluate:
        opt.model = opt.model_eval
        opt.name = opt.name_eval
        if opt.direction == "AtoB":
            opt.direction = "BtoA"
        else:
            opt.direction = "AtoB"

        model_eval = create_model(opt)      # create a model given opt.model and other options
        model_eval.setup(opt)               # regular setup: load and print networks; create schedulers

    MSE_GAN = []
    MSE_seg = []
    for i, data_eval in enumerate(dataset_eval):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model_eval.set_input(data_eval)
        model_eval.test()

        visuals_eval = model_eval.get_current_visuals()
        visuals_eval["original_A"] = data_eval["original_A"] 
        img_path_eval = model_eval.get_image_paths()

        pred_label = model_eval.netG(data_eval["original_A"])
        visuals_eval["predicted_origninal_label"] = pred_label

        MSE_error_seg = torch.mean((pred_label.detach() - visuals["real_A"])**2).cpu().numpy()
        MSE_error_GAN = torch.mean((visuals_eval['output'] - visuals["real_A"])**2).cpu().numpy()
        MSE_GAN.append(MSE_error_GAN)
        MSE_seg.append(MSE_error_seg)
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path_eval))
        save_images(webpage, visuals_eval, img_path_eval, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
    score, variance = inception_score(imgs, resize=True, splits=5)
    print("Inception score", score, variance)
    score, variance = inception_score(real_imgs, resize=True, splits=5)
    print("Inception score data set", score, variance)
    print("MSE of segmentation network on real data: ", np.mean(MSE_error_seg))
    print("MSE of segmentation network on generated data: ", np.mean(MSE_error_GAN))





