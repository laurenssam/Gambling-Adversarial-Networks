
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
from cityscapes import evaluate
import torchvision
import cv2
import scipy.misc
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES']='3'
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

        elif binary_gt.sum() == 0 or binary_pred.sum() == 0:
            f_scores.append(0)
    return np.mean(f_scores), np.mean(house_dist)
        # scipy.misc.imsave("test" + str(class_id) + str(class_id) + ".png", border_gt)
def to_one_hot(labels, C):
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.cpu(), 1)
    return target.cuda().float()


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
       
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # val_set = cityscapes.CityScapes('fine', opt.phase, joint_transform=cityscapes.val_joint_transform, transform=cityscapes.input_transform,
    #                                 target_transform=cityscapes.target_transform)
    # dataset = DataLoader(val_set, batch_size=1, num_workers=8, shuffle=False)
    train_set = cityscapes.CityScapes('fine', opt.phase, joint_transform=cityscapes.val_joint_transform,
                                      transform=cityscapes.input_transform, target_transform=cityscapes.target_transform)
    dataset = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=False)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, "evaluate_" + opt.name + opt.suffix2, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % ("evaluate_" + opt.name, opt.phase, opt.epoch))
    if opt.gambler:
        opt.netD = "unet"
        if opt.argmax:
            netD = networks.define_D(opt.input_nc + opt.output_nc + 1, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
        else:
            netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
        state_dict = torch.load("checkpoints/" + opt.name + "/latest_net_D.pth", map_location="cuda")
        netD.load_state_dict(state_dict)
        netD.eval()
        histogram = []
        layers = []
        counter = 1
        for layer in netD.model.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.LeakyReLU) or isinstance(layer, torch.nn.ConvTranspose2d) or isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.ReLU):
                layers.append(layer)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
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
    
    model.eval()
    for i, data2 in enumerate(dataset):
        data["A"] = data2[0].cuda()
        data["B"] = data2[1].cuda()
        data["A_paths"] = '%04d.png' % i
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        prediction = torch.argmax(model.output, dim=1)
        prediction[(model.data_B == 255)] = 255
        if opt.gambler:
            mask = (data["B"] != 255).unsqueeze(dim=1).expand(model.output.shape).float()
            one_hot_fake = to_one_hot(torch.argmax(model.output, dim=1).unsqueeze(dim=1), 19)
            max_indices = one_hot_fake * 10 
            soft_pred = softmax(model.output + max_indices) * mask
            max_pred = torch.max(soft_pred, dim=1)[0]
            inval = np.linspace(0, 1, 11)
            
            if opt.visualize_features:
                counter = 1
                input_softmax = torch.cat((data["A"], softmax(model.output) * mask), dim=1)
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
                # print(input_softmax.shape)
            softmax_peaked = soft_pred * mask
            softmax_peaked[1-mask.byte()] = -1
            input_softmax_peaked = torch.cat((data["A"], softmax_peaked ), dim=1)
            argmax = to_one_hot(torch.argmax(model.output, dim=1).unsqueeze(dim=1), 19)
            argmax[1-mask.byte()] = -1
            input_argmax = torch.cat((data["A"], argmax), dim=1) 
            input_GT = data["B"].clone()
            input_GT[input_GT == 255] = 0
            input_GT = to_one_hot(input_GT.unsqueeze(dim=1), 19)
            input_GT[1-mask.byte()] = -1
            input_GT = torch.cat((data["A"], input_GT), dim=1)
            input_RGB = torch.cat((data["A"], torch.zeros(softmax_peaked.shape).cuda()), dim=1)
            input_pred_only = torch.cat((torch.zeros(data["A"].shape).cuda(), softmax_peaked), dim=1)
            # one_hot_real = data["B"].clone()
            # one_hot_real[one_hot_real == 255] = 0
            # one_hot_real = to_one_hot(one_hot_real.unsqueeze(dim=1), 19)
            # perfect_structure = (model.output * one_hot_real) + max_indices
            # perfect_structure[(perfect_structure > 0) * (perfect_structure < 0.1)] = 0.1
            # max_perfect_sturecture =  ((1 - torch.max(perfect_structure, dim=1)[0])/18).expand(perfect_structure.shape) * (1 - one_hot_real)
            # soft_pred = perfect_structure + max_perfect_sturecture

            # zero_prediction = torch.cat((torch.zeros(data["A"].shape).cuda(), soft_pred), dim=1)
            # rgb_prediction = torch.cat((data["A"], soft_pred), dim=1)
            # argmax_prediction = torch.cat((torch.zeros(data["A"].shape).cuda(), torch.zeros(soft_pred.shape).cuda()), dim=1)
            # rgb_only = torch.cat((data["A"], torch.zeros(soft_pred.shape).cuda()), dim=1)
            # if opt.argmax:
            #     prediction_only = torch.cat((zero_prediction, torch.zeros(torch.argmax(model.output, dim=1).unsqueeze(dim=1).shape).cuda()), dim=1)
            #     zero_prediction = torch.cat((zero_prediction, torch.argmax(model.output, dim=1).unsqueeze(dim=1).float()), dim=1)
            #     rgb_pred_no_arg = torch.cat((rgb_prediction, torch.zeros(torch.argmax(model.output, dim=1).unsqueeze(dim=1).shape).cuda()), dim=1)
            #     rgb_prediction = torch.cat((rgb_prediction, torch.argmax(model.output, dim=1).unsqueeze(dim=1).float()), dim=1)
            #     argmax_prediction = torch.cat((argmax_prediction, torch.argmax(model.output, dim=1).unsqueeze(dim=1).float()), dim=1)
            #     rgb_only = torch.cat((rgb_only, torch.zeros(torch.argmax(model.output, dim=1).unsqueeze(dim=1).shape).cuda()), dim=1)

            

            # if not torch.equal(torch.argmax(perfect_structure, dim=1), torch.argmax(one_hot_real, dim=1)):
            #     print("Argmax is not matching")
            #     print(torch.sum(torch.argmax(perfect_structure, dim=1) == torch.argmax(one_hot_real, dim=1)))

            # # print(torch.sum(perfect_structure, dim=1))
            # # print(perfect_structure[0, :, 0, 0])
            # # print(one_hot_real[0, :, 0, 0])
            # # print(perfect_structure[0, :,22, 59])
            # # print(one_hot_real[0, :, 22, 59])
            # # quit()
            # rgb_prediction = torch.cat((data["A"], perfect_structure), dim=1)
            # zero_prediction = torch.cat((torch.zeros(data["A"].shape).cuda(), perfect_structure), dim=1)
            with torch.no_grad():
                softmax_output =  softmax(model.output)
                softmax_output[1-mask.byte()] = -1
                input_softmax = torch.cat((data["A"], softmax_output), dim=1)
                bet_map_softmax = sigmoid(netD(input_softmax)).squeeze(dim=1)# + 0.02# N X 1 X 512 X 512, training discr
                # bet_map_softmax = bet_map_softmax * (1/(torch.sum(bet_map_softmax.reshape(bet_map_softmax.shape[0], -1), dim=1)).reshape(bet_map_softmax.shape[0], 1, 1).expand(bet_map_softmax.shape)) * 3000
                
                bet_map_softmax_peaked = sigmoid(netD(input_softmax_peaked)).squeeze(dim=1) #+ 0.02# N X 1 X 512 X 512, training discr
                # bet_map_softmax_peaked = bet_map_softmax_peaked * (1/(torch.sum(bet_map_softmax_peaked.reshape(bet_map_softmax_peaked.shape[0], -1), dim=1)).reshape(bet_map_softmax_peaked.shape[0], 1, 1).expand(bet_map_softmax_peaked.shape)) * 3000
                
                bet_map_argmax = sigmoid(netD(input_argmax)).squeeze(dim=1) #+ 0.02# N X 1 X 512 X 512, training discr
                # bet_map_argmax = bet_map_argmax * (1/(torch.sum(bet_map_argmax.reshape(bet_map_argmax.shape[0], -1), dim=1)).reshape(bet_map_argmax.shape[0], 1, 1).expand(bet_map_argmax.shape)) * 3000
                bet_map_GT = sigmoid(netD(input_GT)).squeeze(dim=1) #+ 0.02# N X 1 X 512 X 512, training discr
                bet_map_RGB = sigmoid(netD(input_RGB)).squeeze(dim=1)
                bet_map_pred_only = sigmoid(netD(input_pred_only)).squeeze(dim=1)
            error_map = (torch.argmax(softmax_peaked.detach(), dim=1) != data["B"]).float() * (data["B"] != 255).float()

            #     betting_map_zero = sigmoid(netD(zero_prediction.detach())).squeeze(dim=1)  # N X 1 X 512 X 512, training discr
            #     # betting_map_zero = betting_map_zero * (1/(torch.sum(betting_map_zero.reshape(betting_map_zero.shape[0], -1), dim=1)).reshape(betting_map_zero.shape[0], 1, 1).expand(betting_map_zero.shape)) * 3000

            #     betting_map_rgb = sigmoid(netD(rgb_prediction.detach())).squeeze(dim=1)  # N X 1 X 512 X 512, training discr
            #     # betting_map_rgb = betting_map_rgb * (1/(torch.sum(betting_map_rgb.reshape(betting_map_rgb.shape[0], -1), dim=1)).reshape(betting_map_rgb.shape[0], 1, 1).expand(betting_map_rgb.shape)) * 3000

            #     betting_map_argmax = sigmoid(netD(argmax_prediction.detach())).squeeze(dim=1)  # N X 1 X 512 X 512, training discr
            #     betting_map_rgb_only = sigmoid(netD(rgb_only.detach())).squeeze(dim=1)  # N X 1 X 512 X 512, training discr
            #     betting_map_prediction_only = sigmoid(netD(prediction_only.detach())).squeeze(dim=1)  # N X 1 X 512 X 512, training discr
            #     betting_map_rgb_pred_no_arg = sigmoid(netD(rgb_pred_no_arg.detach())).squeeze(dim=1) 
                # betting_map_rgb = betting_map_rgb * (1/(torch.sum(betting_map_rgb.reshape(betting_map_rgb.shape[0], -1), dim=1)).reshape(betting_map_rgb.shape[0], 1, 1).expand(betting_map_rgb.shape)) * 3000
            # temp_hist = []
            # for j in range(len(inval) -1):
            #     mask = ((max_pred > inval[j]) * (max_pred < inval[j + 1]))
            #     if torch.sum(mask) == 0:
            #         temp_hist.append(0)
            #     else:
            #         mean_pred_inter = torch.mean(betting_map_rgb[mask])

            #         # print("Mean prediction for interval " +  str(inval[i]) + " and" + str(inval[i + 1]) + ": ", mean_pred_inter.cpu().numpy())
            #         # print("Number of occurences: ", torch.sum(mask.float()).cpu().numpy())
            #         # print("-" * 20)
            #         temp_hist.append(mean_pred_inter.cpu().numpy())
            # histogram.append(temp_hist)
            
            # for name, image in [("all", betting_map_rgb), ("rgb_pred", betting_map_rgb_pred_no_arg), ("pred_arg", betting_map_zero), ("argmax_only", betting_map_argmax), ("rgb_only", betting_map_rgb_only), ("prediction_only", betting_map_prediction_only)]:
            
            bet_maps = [("pred_only", bet_map_pred_only), ("rgb_only", bet_map_RGB), ("softmax", bet_map_softmax), ("softmax_peaked", bet_map_softmax_peaked), ("argmax", bet_map_argmax), ("GT", bet_map_GT), ("error_map", error_map)]
            fig, ax = plt.subplots(nrows=4, ncols=2)
            counter = 0
            for row in ax:
                for col in row:
                    if counter >= len(bet_maps):
                        col.set_visible(False)
                    else:
                        temp = col.imshow(bet_maps[counter][1].squeeze().cpu().numpy(), cmap = 'gray', aspect="auto")
                        col.set_title(bet_maps[counter][0])
                        col.axis("off")
                        counter += 1                
            plt.savefig(web_dir + "/" + str(i) + "betting_maps.png", dpi=200)
            plt.tight_layout()
            plt.close()
        if opt.structure:
            bf, hd = compute_BF(prediction.squeeze().cpu(), model.data_B.squeeze().cpu())
            BF_scores.append(bf)
            hd_dist.append(hd)
        visuals["output"] = prediction
        predictions.append(prediction.squeeze().cpu())
        labels.append(model.data_B.squeeze().cpu())

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML    score, variance = inception_score(real_imgs, resize=True, splits=5)
    acc, acc_cls, mean_iu, iu, _ = evaluate(torch.stack(predictions).numpy(), torch.stack(labels).numpy(), cityscapes.num_classes)

    print("accuracy: ", acc)
    print("class accuracy: ", acc_cls)
    print("IoU: ", mean_iu)
    print("BF score: ", np.mean(BF_scores))
    print("Hausdorff distance: ", np.mean(hd_dist))
    classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
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





