import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle 
# models = [("seg", "CE"), ("pix2pix", "CE + adv"), ("el", "EL-GAN"), ("GambnetsV1", "Gambnets")]
seg = "cityscapes_segementation_no_flip_lrG00001_lrD00002_D1_L3_G-D50-100_ndf64_lamb10_GT00"
focal = "cityscapes_segementation_flip_focal_lrG00001_lrD00002_D1_L3_G-D50-100_ndf64_lamb10_GT00"
ce = "exploit4_scratch_lrG00001_lrD00001_D1_L3_G-D200-400_ndf64_lamb025_GT00"
betting = "u_net_scratch_lrG00001_lrD00001_D1_L3_G-D200-400_ndf64_lamb10_GT00"
HD = "unet_cs_HD_scratch_lrG00001_lrD00005_D2_L3_G-D200-100_ndf128_lamb001_GT00"
el = "el_scratch_lrG00001_lrD00001_D1_L3_G-D200-100_ndf64_lamb001_GT00"
pix = "cs_p2p_scratch_lrG00001_lrD00005_D1_L3_G-D200-100_ndf64_lamb001_GT00"
models = [("seg", seg), ("seg", focal),("pix2pix", pix), ("el", el), ("betting", betting)]

for i in range(len(models)):
	with open("checkpoints/cityscapes/" + models[i][0] + "/" + models[i][1] + "/scores.pickle", 'rb') as handle:
		stats = list(list(zip(*pickle.load(handle)))[2])

	stats = [i * 100 for i in stats]
	# print(list(stats[2])[-20:])
	print(models[i][0])
	print("mean/std: ", np.mean(stats[-20:]), np.std(stats[-20:]))
	print(stats[-20:])
	# print(np.mean([np.std(stats[2][i-5:i]) for i in range(5, len(stats[2]))]))
	# print(np.mean([stats[2][i] - stats[2][i-1] for i in range(1, len(stats[2]))]))
	# print(np.std([stats[2][i] - stats[2][i-1] for i in range(1, len(stats[2]))]))
	n = 10
	print(np.mean([np.mean([stats[i-j] - stats[i-(j + 1)] for j in range(n)]) for i in range(n, len(stats))]))
	print(np.std([np.mean([stats[i-j] - stats[i-(j + 1)] for j in range(n)]) for i in range(n, len(stats))]))

	# print(np.mean([stats[i] - stats[i- 1]for i in range(1, len(stats))]))
	# print(np.std( [stats[i] - stats[i- 1]for i in range(1, len(stats))]))
	positives = 0
	for i in [stats[i] - stats[i- 1]for i in range(1, len(stats))]:
		if i < 0:
			positives += 1
	print(positives)
