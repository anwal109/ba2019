import os.path
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils import data
from Dataset import Dataset

class Net(nn.Module):

	def __init__(self, input_size, size_l1, size_l2, output_size, act_func_name, dropout_prob):
		super(Net, self).__init__()

		act_func = nn.ReLU() if act_func_name == "ReLU" else nn.Tanh() if act_func_name == "tanh" else nn.RReLU() if act_func_name == "PReLU" else nn.Identity()
		dropout = nn.Dropout(p=dropout_prob)
				
		if size_l1 == 0:
			self.fc = nn.Sequential(
				nn.Linear(input_size, output_size)
			)
		elif size_l2 == 0:
			self.fc = nn.Sequential(
				nn.Linear(input_size, size_l1),
				act_func,
				dropout,
				nn.Linear(size_l1, output_size),
			)
		else :
			self.fc = nn.Sequential(
				nn.Linear(input_size, size_l1),
				act_func,
				dropout,
				nn.Linear(size_l1, size_l2),
				act_func,
				dropout,
				nn.Linear(size_l2, output_size),
			)

	def forward(self, x):
		return self.fc(x)

	def train(self, input, target, learn=True):
		output = self.forward(input)

		loss_fn = nn.CrossEntropyLoss()
		loss = loss_fn(output, torch.max(target, 1)[1])

		if learn:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		return loss, [calc_softmax(row) for row in output]


	def test(self, input, target):
		return self.train(input, target, learn=False)
		

	def run(self, input):
		output = self.forward(input)
		return calc_softmax(output)		


def calc_softmax(logits):
	logits = logits.detach().numpy()
	max_logit = max(logits)									# used for normalization
	norm_logits = logits - [max_logit for item in logits]	# stabilizes softmax but keeps output
	e = np.exp(norm_logits)
	dist = e / np.sum(e)
	return dist

def validate(validation_generator, training_rounds, show_wrongs=False, print_console=False):
		losses = []
		probs = []
		corrects = []

		probs_str = ""
		corrects_str = ""
		half_wrongs = ""
		wrongs = ""

		for local_batch, local_labels in validation_generator:
			loss, output = net.test(local_batch, local_labels)
			losses.append(loss.item())
			
			for index, row in enumerate(output):
				if index%type_c == np.argmax(row):
					corrects.append(1)
					corrects_str += "1 "
				elif index_to_type[index] in [index_to_type[top_index] for top_index in row.argsort()[-5:][::-1]]:
					corrects.append(0)
					corrects_str += "0.5 "
					half_wrongs += str(index_to_type[np.argmax(row).item()]) + "-" + str(index_to_type[index]) + "-" + \
									str(round(float(row[np.argmax(row).item()]), 4)) + "-" + str(round(float(row[index]), 4)) + " "
				else:
					corrects.append(0)
					corrects_str += "0 "
					wrongs += str(index_to_type[np.argmax(row).item()]) + "-" + str(index_to_type[index]) + "-" + str(round(float(row[np.argmax(row).item()]), 4)) + "-" + str(round(float(row[index]), 4)) + " "
				
					if show_wrongs:
						print("expected", index_to_type[index], "but got", index_to_type[np.argmax(row).item()])

				probs.append(row[index%type_c])
				probs_str += str(probs[-1]) + " "

		time = str(datetime.now()).split(".")[0]
		with open("scores_tmp/SCORE_" + net_name + ".txt", "a") as score_file:
			score_file.writelines(time + " # " + str(training_rounds) + " # loss: " + str(round(sum(losses)/len(losses), 6)) + " # min: " + str(round(100*min(probs), 2)) + " # prob: " + str(round(100*sum(probs)/len(probs), 2)) + " # max: " + str(round(100*max(probs), 2)) + " # correct: " + str(100*sum(corrects)/len(corrects)) + "\n")

		with open("scores_tmp/SCORES_" + net_name + ".txt", "a") as all_scores:
			all_scores.writelines(time + "#" + str(training_rounds) + "#probs:" + probs_str + "#corrects:" + corrects_str + "\n")

		with open("scores_tmp/WRONGS_" + net_name + ".txt", "a") as all_wrongs:
			all_wrongs.writelines(time + "#" + str(wrongs) + "#" + str(half_wrongs) + "\n")
			
		if print_console:
			print(time, "d_epoch:", training_rounds, "# loss:", round(sum(losses)/len(losses), 4), "#", round(100*min(probs), 2), "-", round(100*sum(probs)/len(probs), 2), "-", round(100*max(probs), 2), "# correct:", round(100*sum(corrects)/len(corrects), 2))



print(str(datetime.now()).split(".")[0], "BEGIN TRAINING:")
print("k:", snakemake.params["ks"], "L1:", snakemake.params["size_l1s"], "L2:", snakemake.params["size_l2s"])
print("act:", snakemake.params["activation_functions"], "drop:", snakemake.params["dropout_probs"], "lr:", snakemake.params["learning_rates"])
print("------------------------------------------------------------------------")

# PARAMS
types = snakemake.params["types"]
type_c = len(types)
index_to_type = {i : int(types[i]) for i in range(type_c)}
training_rounds = snakemake.params["training_rounds"]
validation_rounds = snakemake.params["validation_rounds"]
batch_size = snakemake.params["batch_size"]
trained_nets = []

for k in snakemake.params["ks"]:
	k = int(k)
	input_size = pow(4, k)

	# DATALOADER
	partition = {'train': [t + ".t." + str(t_rnd) for t_rnd in range(training_rounds) for t in types] ,\
			'validation': [t + ".v." + str(v_rnd) for v_rnd in range(validation_rounds) for t in types]}
	labels = {(t+".t."+str(tv_rnd)) if tv_rnd < training_rounds else (t+".v."+str(tv_rnd-training_rounds)) : int(t) for tv_rnd in range(training_rounds+validation_rounds) for t in types}

	params_train = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 32}
	params_val = {'batch_size': type_c, 'shuffle': False, 'num_workers': 32}
	training_set = Dataset(partition['train'], labels, k, types)
	validation_set = Dataset(partition['validation'], labels, k, types)
	training_generator = data.DataLoader(training_set, **params_train)
	validation_generator = data.DataLoader(validation_set, **params_val)

	for size_l1_ratio in snakemake.params["size_l1s"]:
		size_l1 = int(input_size*float(size_l1_ratio))
		for size_l2_ratio in snakemake.params["size_l2s"]:
			size_l2 = int(input_size*float(size_l2_ratio))
			if not (size_l1 == 0 and size_l2 != 0):
				for act_func in snakemake.params["activation_functions"]:
					for dropout_prob in snakemake.params["dropout_probs"]:
						for learning_rate in snakemake.params["learning_rates"]:

							# NET
							net = Net(int(input_size), int(size_l1), int(size_l2), int(type_c), act_func, float(dropout_prob))
							optimizer = optim.Adam(net.parameters(), lr=float(learning_rate))
							net_name = str(type_c) + "_" + str(size_l2) + "_" + str(size_l1) + "_" + str(input_size) + "_" + act_func + "_drop_" + dropout_prob + "_lr_" + learning_rate + "_" + cp

							# INITIALIZATION
							if os.path.isfile("models/MODEL_"+net_name+".pt"):
								checkpoint = torch.load("models/MODEL_"+net_name+".pt")
								net.load_state_dict(checkpoint['state_dict'])
								optimizer.load_state_dict(checkpoint['optimizer'])  
								for param_group in optimizer.param_groups:
									param_group['lr'] = float(learning_rate)
								print(str(datetime.now()).split(".")[0], "Loaded", net_name)
							else :
								print(str(datetime.now()).split(".")[0], "Created", net_name)
								validate(validation_generator, 0)	# Initial validation

							# TRAIN & TEST
							batch_no = 0
							for local_batch, local_labels in training_generator:
								batch_no += 1			# up to (type_c * training_rounds / batch_size) = 100
								_, _ = net.train(local_batch, local_labels)
								if batch_no%25 == 0:	# 25/100 * 128 Training_rounds -> 32 Epochs
									validate(validation_generator, 32, show_wrongs=False, print_console=(batch_no%50 == 0))

							# SAVE NET
							state = {'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
							torch.save(state, "models/MODEL_"+net_name+".pt")
							print(str(datetime.now()).split(".")[0], "Trained", net_name)
							trained_nets.append(net_name)

							print(str(datetime.now()).split(".")[0], "Trained Nets:", len(trained_nets))

with open(snakemake.output["scores"], "a") as out_score:
	out_score.writelines(".")

