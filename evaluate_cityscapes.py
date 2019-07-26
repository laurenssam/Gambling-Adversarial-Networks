
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
from cityscapes import evaluate
import torchvision
import cv2
import scipy.misc
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import voc
import torchvision
import util.transforms as extended_transforms

# print(os.getcwd())

def compute_iou(y_pred, y_true):
     # ytrue, ypred is a flatten vector
     y_pred = y_pred.cpu().numpy().flatten()
     y_true = y_true.cpu().numpy().flatten()
     current = confusion_matrix(y_true, y_pred, labels=np.arange(cityscapes.num_classes))
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     union[union==0] = 1
     IoU = intersection / union.astype(np.float32)
     return np.asarray(IoU), np.mean(IoU)

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

def compute_BF(pred, gt):
    kernel = np.ones((3,3),np.uint8)
    shape_img = gt.shape
    diagonal = np.sqrt(shape_img[0] ** 2 + shape_img[1] ** 2)
    tolerance = 0.0075 * diagonal
    f_scores = []
    house_dist = []
    for class_id in range(cityscapes.num_classes):
        # Create binary maps
        binary_gt = (gt == class_id).float()
        binary_pred = (pred == class_id).float()

        # Only calculate recall and precision when they appear in both the GT and pred, else F-score = 0
        if binary_gt.sum() > 0 and binary_pred.sum() > 0:

            # Dilate and find border
            dilated_gt = cv2.dilate(binary_gt.numpy(), kernel, iterations=1)
            dilated_pred = cv2.dilate(binary_pred.numpy(), kernel, iterations=1)

            border_gt = (torch.from_numpy(dilated_gt) - binary_gt).numpy()
            border_pred = (torch.from_numpy(dilated_pred) - binary_pred).numpy()
   
            # Find the shortest distances 
            distance_gt = scipy.ndimage.morphology.distance_transform_edt(1 - border_gt)# Finds the shortest distance from 1s to 0s
            distance_pred_gt = np.where(border_pred, distance_gt, 100)
            distance_pred = scipy.ndimage.morphology.distance_transform_edt(1 - border_pred)# Finds the shortest distance from 1s to 0s
            distance_gt_pred = np.where(border_gt, distance_pred, 100)

            # Calculate recall and precision
            precision = ((distance_pred_gt < tolerance).sum()/border_pred.sum())
            recall = ((distance_gt_pred < tolerance).sum()/border_gt.sum())

            # Calculate F-score
            if precision == 0 and recall == 0:
                f_scores.append(0)
            else:
                f_score = (2 * precision * recall)/(precision + recall)
                f_scores.append(f_score)

            # Housdorf distance 
            distance_house_dorf = np.nanmean(np.where(border_pred, distance_gt, np.nan))
            distance_house_dorf2 = np.nanmean(np.where(border_gt, distance_pred, np.nan))
            house_dist.append(np.mean([distance_house_dorf, distance_house_dorf2]))
        elif binary_gt.sum() > 0 or binary_pred.sum() > 0:
            f_scores.append(0)
    return np.mean(f_scores), np.mean(house_dist)
        # scipy.misc.imsave("test" + str(class_id) + str(class_id) + ".png", border_gt)
def to_one_hot(labels, C):
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.cpu(), 1)
    return target.cuda().float()

def visualize_feat(model):
    counter = 1
    input_softmax = torch.cat((model.real_A, softmax(model.fake_B) * mask), dim=1)
    store_output = []
    for layer in layers:
        if counter > 17 and isinstance(layer, torch.nn.ConvTranspose2d):
            target_shape = input_softmax.shape
            for output in store_output:
                if output.shape == target_shape:
                    input_softmax = torch.cat((output, input_softmax), dim=1)
                    break
        input_softmax = layer(input_softmax)
        if (isinstance(layer, torch.nn.BatchNorm2d) or counter == 1) and counter < 17:
            store_output.append(input_softmax)

        if not os.path.exists("features/" + str(layer)) and (isinstance(layer, torch.nn.ConvTranspose2d) or isinstance(layer, torch.nn.Conv2d)):
            os.mkdir("features/" + str(layer))
            kernels = input_softmax.detach().cpu()
            for k in range(input_softmax.shape[1]):
                plt.imshow(kernels[0][k], cmap = 'gray')
                plt.savefig("features/" + str(layer) + "/" + str(k) + ".png")
                plt.close()

        counter += 1

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
        train_set = voc.VOCAug(split='val', augment=False, scales=(1.0, 1.0), flip=False, crop_size=512)
        # val_set = voc.VOC('val', joint_transform=voc.val_joint_transform, transform=voc.input_transform,
        #                             target_transform=voc.target_transform)
        # val_loader = DataLoader(val_set, batch_size=opt.batch_size, num_workers=8, shuffle=False)
        opt.output_nc = 21
        cityscapes.num_classes = 21
        cityscapes.classes = voc.classes
        cityscapes.palette = voc.palette
    elif opt.dataset == "camvid":
        opt.output_nc = 11
        cityscapes.num_classes = 11
        cityscapes.classes = camvid.classes
        cityscapes.palette = camvid.palette
        train_set = camvid.Camvid(opt.phase, joint_transform=camvid.val_joint_transform, transform=camvid.input_transform,
                                    target_transform=camvid.target_transform)
        restore_transform = torchvision.transforms.Compose([
        extended_transforms.DeNormalize(*camvid.mean_std),
        ])

    else:
        train_set = cityscapes.CityScapes('fine', opt.phase, joint_transform=cityscapes.val_joint_transform,
                                          transform=cityscapes.input_transform, target_transform=cityscapes.target_transform)
        restore_transform = torchvision.transforms.Compose([
        extended_transforms.DeNormalize(*cityscapes.mean_std),
        ])
    dataset = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=False)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    model.ignore = 255
    web_dir = os.path.join(opt.results_dir, "evaluate_" + opt.name + opt.suffix2, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % ("evaluate_" + opt.name, opt.phase, opt.epoch))
    if opt.gambler:
        
        state_dict = torch.load("checkpoints/" + opt.name + "/latest_net_D.pth", map_location="cuda")
        model.netD.load_state_dict(state_dict)
        model.netD.eval()
        histogram = []
        layers = []
        counter = 1
        # for layer in model.netD.model.modules():
        #     if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.LeakyReLU) or isinstance(layer, torch.nn.ConvTranspose2d) or isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.ReLU):
        #         layers.append(layer)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    # state_dict = torch.load("checkpoints/" + opt.name + "/75_net_G.pth", map_location="cuda")
    # model.netG.load_state_dict(state_dict)
    # model.netG.eval()
    # print("Loaded 75 model")
    dataset_eval = []
    mIoU = []
    IoU_class = []
    accuracies = []
    data = {}
    predictions = []
    labels = []
    BF_scores = []
    hd_dist = []
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=1)

    # Visualize conv filter
    model.visual_names = ["real_A", "real_B"]
    count_cycle =0
    bf_scores_classes = []
    
    model.eval()
    for i, data2 in enumerate(dataset):
        data["A"] = data2[0].cuda()
        data["B"] = data2[1].cuda()
        # data["C"] = data2[2].cuda()
        data["A_paths"] = '%04d.png' % i
        model.set_input(data)  # unpack data from data loader
        with torch.no_grad():
            model.forward()
        visuals = model.get_current_visuals()  # get image results

        visuals["real_A_denorm"] = restore_transform(model.real_A.cpu().squeeze()).unsqueeze(dim=0)
        img_path = model.get_image_paths()     # get image paths
        prediction = torch.argmax(model.fake_B, dim=1)
        prediction[(model.real_B == 255)] = 255
        if opt.gambler:
            mask = (model.real_B != 255).unsqueeze(dim=1).expand(model.fake_B.shape).float()
            one_hot_fake = to_one_hot(torch.argmax(model.fake_B, dim=1).unsqueeze(dim=1), opt.output_nc)
            max_indices = one_hot_fake * 3 
            soft_pred = softmax(model.fake_B + max_indices) * mask
            print("Mean max/Mean min: ", torch.mean(torch.max(soft_pred, dim=1)[0]).item(), torch.mean(torch.min(soft_pred, dim=1)[0]).item())
            print("std max/std min: ",torch.std(torch.max(soft_pred, dim=1)[0]).item(), torch.std(torch.min(soft_pred, dim=1)[0]).item())
            if opt.visualize_features:
                visualize_feat(model)
            with torch.no_grad():
                bet_map_pred_only =model.forward_D(softmax(model.fake_B.detach() + max_indices), zero_rgb=True)[0].detach()
                bet_map_softmax = model.forward_D(softmax(model.fake_B.detach()))[0].detach()
                bet_map_softmax_peaked = model.forward_D(softmax(model.fake_B.detach() + max_indices))[0].detach()
                bet_map_argmax = model.forward_D(one_hot_fake.detach())[0].detach()
                error_map = model.errors

                bet_map_RGB = model.forward_D(torch.zeros(one_hot_fake.shape).to(model.device))[0].detach()
                L1 = np.mean(np.absolute(sigmoid(bet_map_argmax).cpu().numpy() - sigmoid(bet_map_pred_only).cpu().numpy()))
                print("Mean L1 difference prediction only and RGB + prediction:", L1)

            input_GT = model.real_B.clone()
            input_GT[input_GT == 255] = 0
            input_GT = to_one_hot(input_GT.unsqueeze(dim=1), opt.output_nc)
            bet_map_GT = model.forward_D(input_GT.detach())[0].detach()

            bet_maps = [("pred_only", bet_map_pred_only), ("rgb_only", bet_map_RGB), ("softmax", bet_map_softmax), ("argmax", bet_map_argmax), ("softmax_peaked", bet_map_softmax_peaked), ("GT", bet_map_GT), ("error_map", error_map)]
            fig, ax = plt.subplots(nrows=4, ncols=2)
            counter = 0
            for row in ax:
                for col in row:
                    if counter >= len(bet_maps):
                        col.set_visible(False)
                    else:
                        temp = col.imshow(sigmoid(bet_maps[counter][1]).detach().squeeze().cpu().numpy(), cmap = 'gray', aspect="auto")
                        col.set_title(bet_maps[counter][0])
                        col.axis("off")
                        counter += 1                
            plt.savefig(web_dir + "/" + str(i) + "betting_maps.png", dpi=200)
            plt.tight_layout()
            plt.close()
        if opt.structure:
            bf, hd = compute_BF(prediction.squeeze().cpu(), model.real_B.squeeze().cpu())
            BF_scores.append(bf)
            hd_dist.append(hd)
            # bf_scores_classes.append(per_class_bf)
        visuals["output"] = prediction
        if opt.dataset == "voc":
            border = torch.ones(1, 512, 512).cuda().long() * 255
            border_GT = torch.ones(1, 512, 512).cuda().long() * 255
            _ , width, height = prediction.shape
            border[0, 0:width, 0:height] = prediction
            border_GT[0, 0:width, 0:height] = model.real_B
            prediction = border.clone()
            model.real_B = border_GT.clone()
        predictions.append(prediction.squeeze().cpu())
        labels.append(model.real_B.squeeze().cpu())
        count_cycle += torch.sum(model.real_B == 10)

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML    score, variance = inception_score(real_imgs, resize=True, splits=5)
    classes = cityscapes.classes
    if opt.dataset == "camvid":
        del classes[-1]

    if opt.structure:
        print("BF score: ", np.nanmean(BF_scores), np.nanstd(BF_scores))
        print("Hausdorff distance: ", np.mean(hd_dist), np.std(hd_dist), max(hd_dist), min(hd_dist))
        # bf_scores_classes = list(zip(*bf_scores_classes))
        # print(len(bf_scores_classes), len(bf_scores_classes[0]))
        # for i, class_name in enumerate(classes):
        #     print("BF for class " + class_name + ": " + str(np.nanmean(bf_scores_classes[i])))
        # print("MeanBF2: ", np.mean([np.nanmean(i) for i in bf_scores_classes]))
    acc, acc_cls, mean_iu, iu, _ = evaluate(torch.stack(predictions).numpy(), torch.stack(labels).numpy(), cityscapes.num_classes)
    print("NUmber of cycle pixels: ", count_cycle)
    print("accuracy: ", acc)
    print("class accuracy: ", acc_cls)
    print("IoU: ", mean_iu)
    for i, class_name in enumerate(classes):
        print("IoU for class " + class_name + ": " + str(iu[i]))



    # if opt.gambler:
    #     results = list(zip(*histogram))
    #     print(len(results))
    #     fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15,15))
    #     counter = 0
    #     # img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s_prediction.png' % (epoch, iteration + 1, trainer))
    #     for row in ax:
    #         for col in row:
    #             if counter > len(results):
    #                 col.set_visible(False)
    #             else:
    #                 temp = col.hist(list(results[counter]), bins=20)
    #                 col.set_title(str(inval[counter]) + " - " + str(inval[counter + 1]))
    #                 counter += 1
    #     plt.savefig("histograms", bbox_inches="tight", dpi=200)
    #     plt.close()




    # for i, data_eval in enumerate(dataset_eval):
    #     if i >= opt.num_test:  # only apply our model to opt.num_test images.
    #         break
    #     model_eval.set_input(data_eval)
    #     model_eval.test()
    #     visuals_eval = model_eval.get_current_visuals()
    #     visuals_eval["original_A"] = data_eval["original_A"] 
    #     img_path_eval = model_eval.get_image_paths()
    #     pred_label = torch.argmax(model_eval.netG(data_eval["original_A"].cuda()),dim=1)
    #     visuals_eval["predicted_origninal_label"] = pred_label.unsqueeze(dim=1)
    #     visuals_eval['output'] = torch.argmax(visuals_eval['output'], dim=1).unsqueeze(dim=1)
    #     visuals_eval["True_label"] = next(iterator)["B"].unsqueeze(dim=1)
    #     IoU_data, IoU_mean_data = compute_iou(visuals_eval["predicted_origninal_label"], visuals_eval["True_label"])
    #     acc_data = per_pixel_acc(visuals_eval["predicted_origninal_label"], visuals_eval["True_label"])

    #     acc_real.append(acc_data)
    #     IoU_real.append(IoU_mean_data)

    #     IoU_gen, IoU_mean_gen = compute_iou(visuals_eval["output"], visuals_eval["True_label"])
    #     acc_gen = per_pixel_acc(visuals_eval["output"], visuals_eval["True_label"])
    #     acc_fake.append(acc_gen)
    #     IoU_fake.append(IoU_mean_gen)


  
    # print("Per pixel accuracy and IoU on real data: ", np.mean(acc_real), np.mean(IoU_real))
    # print("Per pixel accuracy and IoU on fake data: ", np.mean(acc_fake), np.mean(IoU_fake))





