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
import matplotlib.pyplot as plt
import torchvision
import scipy.misc
import scipy.ndimage
from util import util

import pickle



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
classes = ['Sky','Building','Column-Pole','Road','Sidewalk','Tree','Sign-Symbol','Fence','Car','Pedestrain','Bicyclist','Void']
ignore_label = 255
# root = '/home/samsonl/Documents/cityscapes/'
root = "/media/deepstorage01/datasets_external/camvid/camvid_original/"

# mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
mean_std = ([0.41189489566336, 0.4251328133025, 0.4326707089857], [0.27413549931506, 0.28506257482912, 0.28284674400252])

short_size = int(min(args['input_size']) / 0.5)

palette = [128, 128, 128, 128, 0, 0, 192, 192, 128, 128, 64, 128, 0, 0, 192, 128, 128, 0, 192, 128, 128, 64 ,64, 128, 64, 0, 128, 64, 64, 0, 0, 128, 192, 0, 0, 0]
zero_pad = 256 * 3 - len(palette)

CAMVID_CLASS_COLORS = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (0, 0, 192),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
    (0, 0, 0),
]
for i in range(zero_pad):
    palette.append(0)

train_joint_transform = joint_transforms.Compose([
        joint_transforms.RandomCrop(args['input_size']),
        joint_transforms.RandomHorizontallyFlip()
    ])
# 
val_joint_transform = joint_transforms.Compose([
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
    
def to_one_hot(labels, C, cpu=False):
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.cpu(), 1)
    if cpu:
        return target.float()
    return target.cuda().float()  

# def get_borders(image,  thickness):
#     size = image.shape
#     image = to_one_hot(torch.argmax(image, dim=1).unsqueeze(dim=1), 19).float().cpu().numpy()
#     borders = torch.zeros(size[0], size[2], size[3]).cuda()
#     for i in range(size[0]):
#         for j in range(size[1]):
#             borders[i] += torch.from_numpy(image[i, j] - scipy.ndimage.binary_erosion(image[i, j], iterations=thickness).astype(np.int64)).float().cuda()
#     return borders.float()


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

def plot_G(train, val, opt):
    plt.plot(train, label="train")
    plt.plot(val, label="Val")
    plt.title("Generator loss")
    plt.xlabel("epochs")
    plt.legend(loc=2)
    plt.savefig("checkpoints/" + opt.name + "/G_loss.png")
    plt.close()


def make_dataset(mode):
    img_dir_name = 'images'
    label_dir_name = 'labels'
    mask_path = os.path.join(root, label_dir_name, mode)
    mask_postfix = 'L.png'
    img_path = os.path.join(root, img_dir_name, mode)
    # assert os.listdir(img_path) == os.listdir(mask_path)
    items = []
    categories = os.listdir(img_path)
    counter = 0
    # c_items = [name for name in os.listdir(os.path.join(img_path, c))]
    for it in categories:
        item = (os.path.join(img_path, it), os.path.join(mask_path, it.replace(".png", "_L.png")), counter)
        items.append(item)
        counter += 1
    return items

def get_borders(image, thickness=4):
    image[image == 255] = 0
    image = to_one_hot(image.unsqueeze(dim=1), 19, cpu=True).float().cpu().numpy()
    borders = []
    for i in range(image.shape[0]):
        size = image[i].shape # C X H X W
        border = torch.zeros(size[1], size[2])
        for j in range(size[0]):
            border += torch.from_numpy(image[i, j] - scipy.ndimage.binary_erosion(image[i, j], iterations=thickness).astype(np.int64)).float()
        borders.append(border)
    borders = torch.stack(borders)
    return borders.squeeze().float().numpy()

class Camvid(data.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
        img_path, mask_path, counter = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        mask = torch.tensor(np.array(mask))
        total = 0
        label = torch.ones(mask.shape[:-1]) * 255
        for class_id, rgb in enumerate(CAMVID_CLASS_COLORS[:-1]):
            boolean = (torch.sum((mask.long() == torch.tensor(rgb).long()), dim=2) == 3)
            # total += torch.sum(boolean)
            label[boolean] = class_id
        mask_copy = label.numpy()

        filename = 'borders/Camvid/' + str(self.mode) + "/" + str(index) + ".pickle"
        if not os.path.isfile(filename):
            borders = get_borders(torch.from_numpy(mask_copy).unsqueeze(dim=0).long(), thickness=2)
            with open(filename, 'wb') as handle:
                pickle.dump(borders, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filename, 'rb') as handle:
                borders = pickle.load(handle)
        mask = Image.fromarray(mask_copy.astype(np.uint8))
        borders = Image.fromarray(borders.astype(np.uint8))
        if self.joint_transform is not None:
            img, mask, borders = self.joint_transform(img, mask, borders)
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
                borders = self.target_transform(borders)
            return img, mask, borders, counter

    def __len__(self):
        return len(self.imgs)





