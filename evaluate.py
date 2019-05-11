
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


def compute_iou(y_pred, y_true):
     # ytrue, ypred is a flatten vector
     y_pred = y_pred.cpu().numpy().flatten()
     y_true = y_true.cpu().numpy().flatten()
     current = confusion_matrix(y_true, y_pred, labels=np.arange(12))
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     union[union==0] = 1
     IoU = intersection / union.astype(np.float32)
     return IoU, np.mean(IoU)

def per_pixel_acc(y_pred, y_true):
    correct_predictions = (y_pred.cpu() == y_true.cpu().long()).long().sum().numpy()
    shape = y_true.shape
    total_predictions = np.prod(shape)
    per_pixel_acc = correct_predictions/total_predictions
    return per_pixel_acc
    # print("The per pixel accuray is: ", per_pixel_acc)
    # for i in range(12):
    #     num_class = (y_true.long() == i).long().sum()
    #     num_class_pred = (y_pred.long() == i).long().sum()
    #     print("predictions vs label of class " + str(i) + ": " + str(num_class_pred.cpu().numpy()) + " , " + str(num_class.cpu().numpy()))



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    if opt.netG == "global":
        with open('generated_imgs_pix2pixHD.pickle', 'rb') as handle:
            gen_imgs = pickle.load(handle)
       
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, "evaluate_" + opt.name + opt.suffix2, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
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


        data_eval = data.copy()
        if opt.netG == "global":
            data_eval["A"] = gen_imgs[i]
            imgs.append(gen_imgs[i].squeeze())
        else:
            model.test()           # run inference
            data_eval["A"] = model.fake_B.clone()
            imgs.append(model.fake_B.clone().cpu().squeeze().numpy())
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
        real_imgs.append(model.real_B.clone().cpu().squeeze().numpy())
        data_eval["original_A"] = data["A"].clone()

        dataset_eval.append(data_eval.copy())


    if opt.evaluate:
        opt.model = opt.model_eval
        opt.name = opt.name_eval
        opt.output_nc = 12
        opt.input_nc = 3
        opt.netG = "unet_256"
        opt.dataset_mode = "unaligned"
        if opt.direction == "AtoB":
            opt.direction = "BtoA"
        else:
            opt.direction = "AtoB"
        del dataset
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

        model_eval = create_model(opt)      # create a model given opt.model and other options
        model_eval.setup(opt)               # regular setup: load and print networks; create schedulers

    iterator = iter(dataset)
    acc_real = []
    IoU_real = []

    acc_fake = []
    IoU_fake = []
    for i, data_eval in enumerate(dataset_eval):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model_eval.set_input(data_eval)
        model_eval.test()
        visuals_eval = model_eval.get_current_visuals()
        visuals_eval["original_A"] = data_eval["original_A"] 
        img_path_eval = model_eval.get_image_paths()
        pred_label = torch.argmax(model_eval.netG(data_eval["original_A"].cuda()),dim=1)
        visuals_eval["predicted_origninal_label"] = pred_label.unsqueeze(dim=1)
        visuals_eval['output'] = torch.argmax(visuals_eval['output'], dim=1).unsqueeze(dim=1)
        visuals_eval["True_label"] = next(iterator)["B"].unsqueeze(dim=1)
        IoU_data, IoU_mean_data = compute_iou(visuals_eval["predicted_origninal_label"], visuals_eval["True_label"])
        acc_data = per_pixel_acc(visuals_eval["predicted_origninal_label"], visuals_eval["True_label"])

        acc_real.append(acc_data)
        IoU_real.append(IoU_mean_data)

        IoU_gen, IoU_mean_gen = compute_iou(visuals_eval["output"], visuals_eval["True_label"])
        acc_gen = per_pixel_acc(visuals_eval["output"], visuals_eval["True_label"])
        acc_fake.append(acc_gen)
        IoU_fake.append(IoU_mean_gen)


        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path_eval))
        save_images(webpage, visuals_eval, img_path_eval, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
    score, variance = inception_score(imgs, resize=True, splits=5)
    print("Inception score", score, variance)
    score, variance = inception_score(real_imgs, resize=True, splits=5)
    print("Inception score data set", score, variance)
    print("Per pixel accuracy and IoU on real data: ", np.mean(acc_real), np.mean(IoU_real))
    print("Per pixel accuracy and IoU on fake data: ", np.mean(acc_fake), np.mean(IoU_fake))





