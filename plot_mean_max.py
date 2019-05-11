import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle 
models = ["pix2pix", "EL", "pix2pixHD", "betting"]
for i in range(len(models)):
	with open(models[i] + ".pickle", 'rb') as handle:
		max_prediction = pickle.load(handle)
	mean = np.array([stats[0] for stats in max_prediction])
	std = np.array([stats[1] for stats in max_prediction])
	x = np.linspace(1, len(mean), len(mean))
	plt.plot(x, mean, label=models[i])
	plt.fill_between(x, mean-std, mean+std, alpha=0.4)
	plt.legend()
plt.title("Mean of the max prediction on validation")
plt.ylabel("Max prediction")
plt.xlabel("Epochs")
plt.savefig("temp.png")