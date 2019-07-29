import os
import random
import numpy as np
import torch
from torch.utils import data

class Dataset(data.Dataset):
	def __init__(self, IDs, labels, k, types):
		self.IDs = IDs
		self.labels = labels
		self.k = k
		self.type_c = len(types)
		self.label_to_index = {int(types[i]) : i for i in range(self.type_c)}

	def __len__(self):
		return len(self.IDs)

	def __getitem__(self, index):
		ID = self.IDs[index] 	#001.t.0 // 001.v.0
		
		with open("data/reads/r"+ID+".fq", "r") as in_reads:
			k_meres = []
			# annotation, actual sequence
			_, line = in_reads.readline(), in_reads.readline()[:-1]	
			while line:
				k_meres.append(find_k_meres(self.k, line))
				# +, scores, annotation, actual sequence
				_, _, _, line = in_reads.readline(), in_reads.readline(), in_reads.readline(), in_reads.readline()[:-1]

			k_mere_counts = count_k_meres(self.k, k_meres)
			x = k_mere_counts/np.amax(k_mere_counts)
			
			y = np.zeros(self.type_c)	
			y[self.label_to_index[self.labels[ID]]] = 1

			return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

def find_k_meres(k, seq):
    k_meres = []
    n = len(seq)
    if(k != max(1, min(n, k))):
        k = max(1, min(n, k))
        print('k changed to', k)
    
    for i in range(n-k+1):
        k_meres.append(seq[i:i+k])

    return k_meres  


def count_k_meres(k, k_meres):
	# ['A', 'C', 'G', 'T']
	nt_count = 4
	# can hold all possible k_meres
	k_mere_counts = np.zeros(pow(nt_count, k), dtype=int)
	for k_mer_list in k_meres:
		for k_mer in k_mer_list:
			pos = 0
			# iterate over k-meres and sort into list of all k-meres using a quaternary tree
			for i in range(k):
				if(k_mer[i] == 'C'):
					pos += pow(nt_count, k-i-1)
				elif(k_mer[i] == 'G'):
					pos += pow(nt_count, k-i-1)*2
				elif(k_mer[i] == 'T'):
					pos += pow(nt_count, k-i-1)*3
			k_mere_counts[pos] += 1

	return k_mere_counts


def calc_weighted_k_meres(k, seq):
    k_meres = find_k_meres(k, seq)
    k_mere_counts = count_k_meres(k, k_meres)
    maximum = np.amax(k_mere_counts)
    # normalize all values (between 0 and 1)
    # 0.5 -> half as often as the most common k_mere
    return k_mere_counts/maximum

