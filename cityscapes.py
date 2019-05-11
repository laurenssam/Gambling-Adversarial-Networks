import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
import util.joint_transforms as joint_transforms
import util.transforms as extended_transforms
import torchvision.transforms as standard_transforms
from math import sqrt
import torch
import matplotlib
matplotlib.use('Agg')
import torchvision
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
from util import util



args = {
    'train_batch_size': 16,
    'epoch_num': 500,
    'lr': 1e-10,
    'weight_decay': 5e-4,
    'input_size': (512, 512),
    'momentum': 0.95,
    'lr_patience': 100,  # large patience denotes fixed lr
    'snapshot': '',  # empty string denotes no snapshot
    'print_freq': 20,
    'val_batch_size': 16,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.05  # randomly sample some validation results to display
}
classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "trafficlight", "trafficsign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
num_classes = 19
ignore_label = 255
# root = '/home/samsonl/Documents/cityscapes/'
root = "/media/deepstorage01/datasets_external/cityscapes/cityscapes_thesis/"

# mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
mean_std = ([0.3006, 0.3365, 0.2956], [0.1951, 0.1972, 0.1968])

short_size = int(min(args['input_size']) / 0.5)

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

train_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(short_size),
        joint_transforms.RandomCrop(args['input_size']),
        joint_transforms.RandomHorizontallyFlip()
    ])
# 
val_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(short_size),
        # joint_transforms.CenterCrop(args['input_size']),
    ])
input_transform_train = standard_transforms.Compose([
        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
target_transform = extended_transforms.MaskToTensor()
restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
])
visualize = standard_transforms.ToTensor()


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    # for i in range(num_classes):
    #     print((gts == i).sum())
    #     print((predictions == i).sum())
    #     print(i)
    #     print("-" * 20)
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, iu, fwavacc

def plot_stats(stats, iu_class, name):
    labels = ["acc", "acc_cls", "IoU", "train_acc", "train_acc_cls", "train_IoU"]
    counter = 0
    for temp in zip(*stats):
        plt.plot(temp - temp[0], label=labels[counter])
        counter += 1
    plt.title("stats")
    plt.xlabel("epochs")
    plt.legend(loc=2)
    plt.savefig("checkpoints/" + name + "/stats.png")
    plt.close()

    # High frequency plot
    classes_iu = list(zip(*iu_class))
    high_freq_ids = [4, 5, 6, 7, 11, 12, 13, 17, 18]
    low_freq_ids = [0, 1, 2, 3, 8, 9, 10, 14, 15, 16]
    for ids in low_freq_ids:
        plt.plot(classes_iu[ids] - classes_iu[ids][0], label=classes[ids])
    plt.title("IoU per class (Low Frequency classes)")
    plt.legend(loc=2)
    plt.savefig("checkpoints/" + name + "/lowfrequency.png")
    plt.close()

    for ids in high_freq_ids:
        plt.plot(classes_iu[ids] - classes_iu[ids][0], label=classes[ids])
    plt.title("IoU per class (High Frequency classes)")
    plt.legend(loc=2)
    plt.savefig("checkpoints/" + name + "/highfrequency.png")
    plt.close()

def calculate_weight_vector(train_loader):
    counts = torch.zeros(num_classes)
    count_images = torch.zeros(num_classes)
    for i, data in enumerate(train_loader):
        print(i)
        for class_id in range(num_classes):
            count = (data[1]==class_id).float().sum()
            if count > 0: 
                count_images[class_id] += 1
                counts[class_id] += count
    freq = np.divide(counts, count_images)
    prob = (counts/sum(counts))
    print(prob)
    weight_mean_freq = prob**(-1)
    print(np.round(weight_mean_freq, decimals=2))
    w_class = 1/(np.log(1.02 + prob))
    print(np.round(w_class, decimals=2))
    w_median = np.median(freq)/freq
    print(np.round(w_median, decimals=2))
    return None
    
def to_one_hot(labels, C):
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.cpu(), 1)
    return target.cuda().float()  

def get_borders(image,  thickness):
    size = image.shape
    image = to_one_hot(torch.argmax(image, dim=1).unsqueeze(dim=1), 19).float().cpu().numpy()
    borders = torch.zeros(size[0], size[2], size[3]).cuda()
    for i in range(size[0]):
        for j in range(size[1]):
            borders[i] += torch.from_numpy(image[i, j] - scipy.ndimage.binary_erosion(image[i, j], iterations=thickness).astype(np.int64)).float().cuda()
    return borders.float()

def calculate_stats(data_loader, model, name, opt, val=True, gambler=0):
    if not os.path.isdir('checkpoints/' + name + "/validation_imgs"):
        os.mkdir('checkpoints/' + name + "/validation_imgs")
        print("Created folder for validation imgs")
    if not os.path.isdir('checkpoints/' + name + "/validation_imgs_gambler") and gambler:
        os.mkdir('checkpoints/' + name + "/validation_imgs_gambler")
    labels = []
    predictions = [] 
    mean_max = []
    max_stats = []
    gambler_loss = []
    gambler_predictions = []
    errors = []
    model.netG.eval()
    model.netD.eval()
    BCEweight = torch.tensor([opt.weight_gambler]).cuda()
    criterionBCE = torch.nn.BCEWithLogitsLoss(pos_weight=BCEweight, size_average=True)
    softmax = torch.nn.Softmax(dim=1)
    sigmoid = torch.nn.Sigmoid()
    random_imgs = [0, 10, 20, 30, 40] # evaluation
    correct_predictions = 0
    n_errors = 0
    n_positives = 0
    losses_D = []
    average_bet = []
    for i, data in enumerate(data_loader):  # inner loop within one epoch
        data[0] = data[0].cuda()
        data[1] = data[1].cuda()
        with torch.no_grad():
            prediction = model.netG(data[0])
            if gambler:
                mask = (data[1] == 255).unsqueeze(dim=1).expand(prediction.shape) # mask for ignore class, zeroes out the ignore class   
                # max_indices = self.to_one_hot(torch.argmax(self.fake_B.clone().detach(), dim=1).unsqueeze(dim=1), 19) * 100 # N x 512 X 512 --> N X 1 X 512 X 512 --> N X 19 X 512 X 512 --> * 100
                fake_B_masked = to_one_hot(torch.argmax(prediction.clone().detach(), dim=1).unsqueeze(dim=1), 19)
                fake_B_masked[mask] = -1 

                borders = get_borders(fake_B_masked.clone().detach(), opt.thickness) # N X 512 X 512
                fake_AB = torch.cat((data[0], fake_B_masked), 1)  # concat input rgb + masked output

                output_gambler = model.netD(fake_AB.detach()).squeeze()
                error = (torch.argmax(fake_B_masked, dim=1) != data[1]).float() * (data[1]!= 255).float() * (1 - borders)

                loss_D = criterionBCE(output_gambler, error).detach().item()
                bet_map = sigmoid(output_gambler.detach())

                average_bet.append(torch.mean(bet_map).detach().cpu().item())
                positives = (bet_map > 0.5)
                errors = (error == 1)
                n_errors += errors.sum().cpu().item()
                n_positives += positives.sum().cpu().item()
                correct_predictions += (errors * positives).sum().cpu().item()
                losses_D.append(loss_D)
        if val and i in random_imgs:
            im_data = torch.argmax(prediction, dim=1).cpu().data
            im_data[0][(data[1][0] == 255)] = 255
            im = util.tensor2im(im_data)
            save_path = 'checkpoints/' + name + "/validation_imgs/" + "img" + str(i) + "_epoch" + str(model.epoch)+".png"
            util.save_image(im, save_path)
            if gambler:
                save_path = 'checkpoints/' + name + "/validation_imgs_gambler/" + "img" + str(i) + "_epoch" + str(model.epoch)+".png"
                plt.imshow(bet_map[0].squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                plt.savefig(save_path)
                plt.close()
                save_path = 'checkpoints/' + name + "/validation_imgs_gambler/" + "img" + str(i) + "_epoch" + str(model.epoch)+"_GT.png"
                plt.imshow(error[0].squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                plt.savefig(save_path)
                plt.close()
        mean_max.append(torch.mean(torch.max(softmax(prediction), dim=1)[0].cpu()).data)
        predictions.append(torch.argmax(prediction, dim=1).squeeze().cpu()) # N X 512 X 512
        labels.append(data[1].squeeze().cpu()) # N X 512 X 512
    acc, acc_cls, mean_iu, iu, fwavacc = evaluate(torch.cat(predictions, dim=0).numpy(), torch.cat(labels, dim=0).numpy(), num_classes) # N X 512 X 1024 --> D X 512 X 1024
    model.netG.train()
    model.netD.train()
    if val:
        print(20 * "-")
        for i in range(len(iu)):
            print("IoU for class " + classes[i] + " is: " + str(iu[i]))
        print(20 * "-")
        print("Accuray: ", acc)
        print("Class accuracy: ", acc_cls)
        print("Mean IoU: ", mean_iu )
        print("Mean max: ", np.mean(mean_max))
        print("std max: ", np.std(mean_max))
        max_stats = (np.mean(mean_max), np.std(mean_max))
    if gambler:
        print(20 * "-")
        print("GAMBLER STATS")
        print(correct_predictions, n_errors, n_positives)
        if n_positives == 0:
            precision = 0
        else:
            precision = correct_predictions/float(n_positives) # Relevant by number of positives
        if n_errors == 0:
            recall = 0
        else:
            recall = correct_predictions/float(n_errors) # Relevant prediction/number of relevant 
        if (recall + precision) == 0:
            F1_score = 0
        else:
            F1_score = (2 * recall * precision)/(precision + recall)
        print("Precision/Recall/F1: ", precision, recall, F1_score)
        loss_gambler = np.mean(losses_D)
        print("Loss:", loss_gambler)
        print("average bet:", np.mean(average_bet))
        return acc, acc_cls, mean_iu, fwavacc, list(iu), max_stats, [precision, recall, F1_score, loss_gambler]
    return acc, acc_cls, mean_iu, fwavacc, list(iu), max_stats


def plot_gambler(train, val, opt):
    train = list(zip(*train))
    val = list(zip(*val))
    labels = ["Precision", "Recall", "F1", "Loss"]
    for i in range(len(train) - 1):
        plt.plot(train[i], label="train_" + labels[i])
        plt.plot(val[i], label="val" + labels[i])
    plt.title("stats")
    plt.xlabel("epochs")
    plt.legend(loc=2)
    plt.savefig("checkpoints/" + opt.name + "/gambler_stats.png")
    plt.close()

    plt.plot(train[-1], label="train_" + labels[-1])
    plt.plot(val[-1], label="val" + labels[-1])
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend(loc=2)
    plt.savefig("checkpoints/" + opt.name + "/gambler_loss.png")
    plt.close()



def make_dataset(quality, mode):
    assert (quality == 'fine' and mode in ['train', 'val']) or \
           (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])

    if quality == 'coarse':
        img_dir_name = 'leftImg8bit_trainextra' if mode == 'train_extra' else 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(root, 'gtCoarse', 'gtCoarse', mode)
        mask_postfix = '_gtCoarse_labelIds.png'
    else:
        img_dir_name = 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(root, 'gtFine_trainvaltest', 'gtFine', mode)
        mask_postfix = '_gtFine_labelIds.png'
    img_path = os.path.join(root, img_dir_name, 'leftImg8bit', mode)
    assert os.listdir(img_path) == os.listdir(mask_path)
    items = []
    categories = os.listdir(img_path)
    counter = 0
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix), counter)
            items.append(item)
            counter += 1
    return items


class CityScapes(data.Dataset):
    def __init__(self, quality, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(quality, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
        img_path, mask_path, counter = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask, counter

    def __len__(self):
        return len(self.imgs)
