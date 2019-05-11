import cityscapes
import cv2
from torch.utils.data import DataLoader
import numpy as np
import torch
import scipy.misc as sc

train_set = cityscapes.CityScapes('fine', 'train', joint_transform=cityscapes.train_joint_transform,
                                      transform=cityscapes.input_transform, target_transform=cityscapes.target_transform)
train_loader = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=True)
kernel = np.ones((3,3),np.uint8)
frequencies = []
for j, data in enumerate(train_loader):
	if j % 25 == 0 and j > 0:
		print(j)
	label = data[1].squeeze().float() + 1
	# color_label = cityscapes.visualize(cityscapes.colorize_mask(label.numpy()).convert("RGB"))
	frequency_per_label = []
	for i in range(1, cityscapes.num_classes + 1):
		mask = (label == i).float()
		label_proccesed = mask * label
		label_proccesed[label_proccesed > 0] = 255
		dilated = cv2.dilate(label_proccesed.numpy(), kernel, iterations=1)
		dilated_torch = torch.from_numpy(dilated)
		dilated_area = dilated_torch - label_proccesed
		pixels = (label_proccesed > 0).sum()
		pixels_dilated = (dilated_area > 0).sum() 
		# print(cityscapes.classes[i - 1])
		# print("Number of pixels class: ", pixels)
		# print("Number of pixels dilated area: ", pixels_dilated)
		if pixels_dilated > 0:
			frequency = float(pixels_dilated)/float(pixels)
			frequency_per_label.append(frequency)
		else:
			frequency_per_label.append(-1)
			# print("Frequency measure is: ", frequency)
			# print("class is not present")
	frequencies.append(frequency_per_label)
		# print("-" * 20)
		# sc.imsave("dilate" + str(i) + ".png", dilated_area)
		# sc.imsave("label" + str(i) + ".png", label_proccesed)
frequency = list(zip(*frequencies))
mean_list = []
median_list = []
for i, freq in enumerate(frequency):
	print("Class: ", cityscapes.classes[i])
	freq = list(filter(lambda a: a != -1, freq))
	print("Number of occurences of the class: ", len(freq))
	print("mean, median, variance : ", np.mean(freq), np.median(freq), np.var(freq))
	print("min, max: ", min(freq), max(freq))
	print("-" * 20)
	mean_list.append((cityscapes.classes[i], np.mean(freq)))
	median_list.append((cityscapes.classes[i], np.median(freq)))

sorted_mean = sorted(mean_list, key=lambda tup: tup[1])
sorted_median = sorted(median_list, key=lambda tup: tup[1])
print("Sorted list based on mean: " , [value[0] for value in sorted_mean])
print("Sorted list based on median: ", [value[0] for value in sorted_median])