import pandas as pd
import numpy as np
import pickle

def readdata(dataszie):

	df = pd.read_csv("USDJPY.csv")
	price = df["CLOSE"].iloc[:dataszie]
	return price

def cut_quantile(tmp_price, state, quantile_size, cutnum):

	quantiles = []
	for i in range(quantile_size-1):
		tmp_q = np.percentile(tmp_price, cutnum[i])
		quantiles.append(tmp_q)
	return quantiles

def compute_M(tmp_M, tmp_real_price, tmp_num, flag_1, flag_2):

	if flag_1 >= flag_2:

		tmp_flag_real_price = tmp_real_price[flag_2:flag_1+1+1][::-1]
		state = np.zeros((tmp_num, quantile_size-1+4), dtype="float32")
		# print("roll :", flag_1, flag_2)

		state[:,0] = tmp_flag_real_price[:-1]
		state[:,1] = tmp_flag_real_price[1:]
	
		quantiles = cut_quantile(tmp_flag_real_price, state, quantile_size, cutnum=cutposition)

		for i in range(quantile_size-1): # 3 - 1 = 2
			state[:,2+i] = state[:,0] >= quantiles[i]
		state[:,2+quantile_size-1] = state[:,2:2+quantile_size-1].sum(axis=1)

		for i in range(quantile_size-1): # 3 - 1 = 2
			state[:,2+i] = state[:,1] >= quantiles[i]
		state[:,2+quantile_size-1+1] = state[:,2:2+quantile_size-1].sum(axis=1)

		# 10 * 3 * 3
		f_matrix = np.zeros((tmp_num, quantile_size, quantile_size), dtype="int")
		for i in range(tmp_num):
			f_matrix[i, state[i,2+quantile_size-1], state[i,2+quantile_size-1+1]] = 1

		f_matrix_sum = np.sum(f_matrix, axis=0)

		for c in range(quantile_size):
			c_sum = np.sum(f_matrix_sum[:,c])
			for r in range(quantile_size):
				tmp_M[flag_1, flag_2, r, c] = f_matrix_sum[r,c]/c_sum

	elif flag_1 < flag_2:

		tmp_flag_real_price = tmp_real_price[flag_1:flag_2+1+1]
		state = np.zeros((tmp_num, quantile_size-1+4), dtype="float32")
		# print("roll :", flag_1, flag_2)

		state[:,0] = tmp_flag_real_price[:-1]
		state[:,1] = tmp_flag_real_price[1:]
	
		quantiles = cut_quantile(tmp_flag_real_price, state, quantile_size, cutnum=cutposition)

		for i in range(quantile_size-1): # 3 - 1 = 2
			state[:,2+i] = state[:,0] >= quantiles[i]
		state[:,2+quantile_size-1] = state[:,2:2+quantile_size-1].sum(axis=1)

		for i in range(quantile_size-1): # 3 - 1 = 2
			state[:,2+i] = state[:,1] >= quantiles[i]
		state[:,2+quantile_size-1+1] = state[:,2:2+quantile_size-1].sum(axis=1)

		# 10 * 3 * 3
		f_matrix = np.zeros((tmp_num, quantile_size, quantile_size), dtype="int")
		for i in range(tmp_num):
			f_matrix[i, state[i,2+quantile_size-1], state[i,2+quantile_size-1+1]] = 1

		f_matrix_sum = np.sum(f_matrix, axis=0)

		for r in range(quantile_size):
			r_sum = np.sum(f_matrix_sum[:,r])
			for c in range(quantile_size):
				tmp_M[flag_1, flag_2, r, c] = f_matrix_sum[r,c]/r_sum

	return tmp_M

def main():

	price = readdata(1200)

	real_windows_size = windows_size + 1

	# Save features data
	features_matrix = np.zeros((rolling_time, quantile_size * quantile_size, windows_size, windows_size))

	# Save price data
	price_array = np.zeros((rolling_time, windows_size))

	num = 0
	for i_roll in range(rolling_time):

		# if num == 2:
		# 	break

		tmp_real_price = price[i_roll:i_roll + real_windows_size]
		price_array[i_roll, :] = tmp_real_price[:-1]

		tmp_M = np.zeros((windows_size, windows_size,  quantile_size, quantile_size))

		for flag_1 in range(windows_size):
			for flag_2 in range(windows_size):

				tmp_num = np.abs(flag_1 - flag_2) + 1
				if tmp_num >= quantile_size:

					tmp_M = compute_M(tmp_M, tmp_real_price, tmp_num, flag_1, flag_2)

		count = 0
		for r in range(quantile_size):
			for c in range(quantile_size):
				features_matrix[i_roll, count, :, :] = tmp_M[:, :, r, c]
				count += 1

		num += 1
		print("Finished:", num)

	with open('./features_matrix2.pickle', 'wb') as f:
	    pickle.dump({"features_matrix":features_matrix, 
	    	         "price_array":price_array}, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

	windows_size = 100
	quantile_size = 4
	cutposition = [25, 50, 75]
	rolling_time = 100

	#------------------------------------------------------------------------------	
	main()