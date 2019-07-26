import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle 
models = [("seg", "CE"), ("pix2pix", "CE + adv"), ("el", "EL-GAN"), ("pix2pixHD", "MD EL-GAN"), ("GambnetsV1", "Gamb nets"), ("GambnetsV2", "Gamb nets - arg")]
# models = [("seg", "CE"), ("pix2pix", "CE + adv"), ("el", "EL-GAN"), ("GambnetsV1", "Gambnets")]

for i in range(len(models)):
	with open("plots/mean_max/" + models[i][0] + ".pickle", 'rb') as handle:
		max_prediction = pickle.load(handle)
	mean = np.array([stats[0] for stats in max_prediction])
	std = np.array([stats[1] for stats in max_prediction])
	if models[i][0] == "seg":
		x = np.linspace(1, len(mean), len(mean))
		seg_length = len(mean)
		plt.plot(x, mean, label=models[i][1])
		plt.fill_between(x, mean-std, mean+std, alpha=0.4)
	else:
		x = np.linspace(seg_length, seg_length + len(mean), len(mean))
		plt.plot(x[:50], mean[:50], label=models[i][1])
		plt.fill_between(x[:50], mean[:50]-std[:50], mean[:50]+std[:50], alpha=0.4)

	plt.ylim(0.7, 1)

	plt.legend(loc=4)
	print("Mean/std over last 10 epochs", np.mean(mean[-1]), np.mean(std[-1]))

# plt.title("Mean of the max prediction on validation")
plt.ylabel("Mean max prediction")
plt.xlabel("Epochs")
plt.savefig("full_mean_max.png")
plt.close()

# models = ["seg", "pix2pix_scratch", "el_scratch" , "pix2pixHD_scratch", "betting_scratch", ]
for i in range(len(models)):
	file_name = "plots/mean_max/" + models[i][0] + "_scratch"+ ".pickle"
	if models[i][0] == "seg":
		file_name = "plots/mean_max/" + models[i][0] + ".pickle"
	with open(file_name, 'rb') as handle:
		max_prediction = pickle.load(handle)
	mean = np.array([stats[0] for stats in max_prediction])
	std = np.array([stats[1] for stats in max_prediction])
	if models[i][0] == "seg":
		#Y = [10, 20, 40, 80, 110]
		N = len(mean)
		X = np.arange(0, 2*N, 2)
		X_new = np.arange(2*N-1)       # Where you want to interpolate
		mean = np.interp(X_new, X, mean) 
		std = np.interp(X_new, X, std) 
	x = np.linspace(1, len(mean), len(mean))
	plt.ylim(0.7, 1)
	plt.plot(x[:200], mean[:200], label=models[i][1])
	plt.fill_between(x[:200], mean[:200]-std[:200], mean[:200]+std[:200], alpha=0.4)
	plt.legend(loc=4)
	print(models[i][0])
	print("Mean/std over last 10 epochs", np.mean(mean[:200][-5]), np.mean(std[:200][-5]))
# plt.title("Mean of the max prediction on validation")
plt.ylabel("Mean max prediction")
plt.xlabel("Epochs")
plt.savefig("scratch_mean_max.png")
plt.close()