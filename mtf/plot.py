import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import pickle

with open('features_matrix2.pickle', 'rb') as f:
	# 100 * 16 * 100 * 100
    data_dict = pickle.load(f)  

features_matrix = data_dict["features_matrix"]
price_array = data_dict["price_array"]
print(features_matrix.shape)


for t in range(features_matrix.shape[0]):

	# (4*100) * (4*100)
	arr = np.zeros((400,400))
	count = 0
	for r in range(4):
		for c in range(4):
			arr[0+(r*100):100+(r*100), 0+(c*100):100+(c*100)] = features_matrix[t,count,:,:]
			count += 1

	plt.clf()

	fig = plt.figure()
	ax1 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=2)
	ax2 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=2)
	plt.subplots_adjust(top=0.7, bottom=0.3, wspace=0.1)
	ax1.imshow(arr, cmap="nipy_spectral")
	ax2.plot(price_array[t])
	fig.savefig("./images/" + str(t)+".png", dpi=100)
	plt.close("all")

	print("finish :" + str(t))






