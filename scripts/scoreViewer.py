import os
import re
import random
import math
import matplotlib.pyplot as plt		# !! version = 3.0.1 !!

def save(title, size_to_show, net_name):
	net_name = net_name.replace(".", "")
	if title == "losses":
		plt.ylim(bottom=0)
	else:
		plt.ylim(0, 100)
	plt.xlim(left=0)
	plt.grid()
	plt.legend(legend)
	plt.savefig(snakemake.params["path_to_figures"]+title+net_name+".png")
	plt.clf()

def get_spaced_colors(n):
	colors = []
	for i in range(n):
		colors.append([(i+1)/n, (i*(math.pi-1.5))%1, (1-(i+1)*2/n)%1, 1])
	return colors

def decide_correct(corrects):
	if "0" in corrects:
		return "0"
	elif "0.5" in corrects:
		return "0.5"
	else :
		return "1"

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

score_dir = "./scores/"
score_files = os.listdir(score_dir)
score_files.sort(key=natural_keys, reverse=True)



for size_to_show in snakemake.params["show_size_out"]:
	all_epochs = []
	all_probs = []
	all_corrects = []
	legend = []

	for in_score in score_files:
		if in_score[:6] == "SCORES":
			#	[SCORE, 200, 2048, 1024, 64, tanh, drop, 0, lr, 0.001, batch, 256]
			_, size_out, size_l2, size_l1, size_in, act, _, dropout, _, lr = in_score[:-4].split("_")

			if size_out == size_to_show \
				and (len(snakemake.params["show_size_in"]) == 0 or size_in  in snakemake.params["show_size_in"]) \
				and (len(snakemake.params["show_size_l2"]) == 0 or size_l2  in snakemake.params["show_size_l2"]) \
				and (len(snakemake.params["show_size_l1"]) == 0 or size_l1  in snakemake.params["show_size_l1"]) \
				and (len(snakemake.params["show_act"]) 	   == 0 or act 		in snakemake.params["show_act"]) \
				and (len(snakemake.params["show_dropout"]) == 0 or dropout  in snakemake.params["show_dropout"]) \
				and (len(snakemake.params["show_lr"]) 	   == 0 or lr 		in snakemake.params["show_lr"]) :
				
				with open(score_dir+in_score, "r") as score:
					line = score.readline()
					epochs = []
					probs = []
					corrects = []
					
					while line and (True if len(epochs)==0 else epochs[-1] < 640):
					# 	2019-07-19 17:02:17#32#probs:0.06327853 0.013720752 0.01063568 0.016760169 0.008311098 0.006334671 0.013319984 0.007632533 0.010809664 0.00825491 0.0058664763 0.013801681 0.0068149706 0.010298105 0.019931845 0.010860624 0.019142333 0.006182926 0.012262218 0.021145366 0.010922623 0.007081737 0.0055113803 0.02459953 0.032793056 0.004807438 0.017642584 0.008648934 0.025219733 0.011931951 0.0084362645 0.013211225 0.013514237 0.035218515 0.0055385055 0.005450837 0.010891142 0.061803296 0.0075749066 0.012311158 0.021506462 0.017651962 0.03205618 0.009128597 0.065946735 0.046549097 0.038543437 0.035129722 0.0066515394 0.010025105 0.010697187 0.005541424 0.0080874115 0.009983253 0.03160384 0.007213183 0.011413318 0.019711012 0.02454648 0.016364219 0.007540931 0.014024491 0.026052589 0.0039468203 0.0037627802 0.022265974 0.0058048866 0.04720969 0.005789812 0.010714549 0.06999491 0.0017460597 0.02932696 0.005812044 0.012080987 0.0035656083 0.037138868 0.029387377 0.021954268 0.047895845 0.037075978 0.009562025 0.01993958 0.0377177 0.009797433 0.016697183 0.06450531 0.013308215 0.0056171203 0.0034191008 0.03615367 0.013353424 0.0146402 0.024308475 0.06007106 0.014455591 0.012216558 0.0053589977 0.0075652 0.03626816 0.008313945 0.12933674 0.0026941355 0.011792506 0.0069773835 0.038411815 0.010425346 0.01131564 0.011839788 0.014385892 0.033639915 0.005611613 0.05855127 0.06886149 0.04029162 0.012290568 0.028475642 0.024288006 0.009584017 0.19907126 0.17512164 0.02340523 0.0096139405 0.021514071 0.01443119 0.016489007 0.020046564 0.017601343 0.017313791 0.0049335216 0.004874406 0.019767303 0.019785853 0.010069009 0.0077965762 0.015000825 0.006828504 0.0193031 0.025672123 0.04601345 0.051127344 0.012749983 0.018895548 0.015018699 0.22669992 0.0076337038 0.010938572 0.13971472 0.043432817 0.1635878 0.011358585 0.005446127 0.0025532276 0.008407744 0.00966122 0.009863989 0.019182356 0.040945917 0.00542263 0.011538173 0.020782162 0.015543164 0.015263531 0.010761977 0.0071429927 0.009148137 0.0030982029 0.011595761 0.027087303 0.031483285 0.031639643 0.055451863 0.019099975 0.011709879 0.044641912 0.010718352 0.015121342 0.035474878 0.013508345 0.5127701 0.0075215143 0.03806874 0.015343444 0.029659426 0.013184944 0.021446683 0.48531434 0.020698618 0.014932445 0.023858787 0.00983017 0.0037108196 0.017128844 0.012645295 0.018006496 0.066672176 0.03183594 0.05302622 0.01913372
					#    0.009455013 #corrects:1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0.5 0 0 0 0 0 0 0.5 1 0 0 0 0 0 0 0.5 0 1 0 0 0 1 0 0 0.5 1 0.5 0 1 0.5 1 1 0 0 0 0 0 0 0.5 0 0 0 0.5 0.5 0 0 0.5 0 0 0.5 0 0.5 0 0 1 0 1 0 0 0 0.5 0 0.5 0.5 0.5 0 0 1 0 0 1 0 0 0 0.5 0 0.5 0 1 0 0 0 0 1 0 1 0 0 0 0.5 0 0 0 0 0 0 1 0.5 0.5 0 0 0 0 1 1 0 0 0 0.5 0.5 0 1 0.5 0 0 0 0 0 0 0 0 0 0.5 1 0.5 0 1 0.5 0.5 0 0 1 0.5 1 0 0 0 0 0 0 1 0.5 0 0 0 0 1 0 0 0 0 0 0.5 0.5 1 1 0.5 0 1 0 1 1 0 1 0 0.5 0.5 0.5 0 0.5 1 0 0 0 0 0 0 0 0.5 0.5 1 1 0.5 0
						performance = line.split("#")
						line_probs = performance[2][6:].split(" ")[:-1]		# [6:] -> "probs:"
						line_corrects = performance[3][9:].split(" ")[:-1]	# [9:] -> "corrects"
						item_c = len(line_probs)
						
						epochs.append(float(performance[1])+epochs[-1] if len(epochs) > 0 else float(performance[1]))
						probs.append(100.0*sum([float(prob) for prob in line_probs])/item_c)
						corrects.append(100.0*sum([1 if item == "1" else 0 for item in line_corrects])/item_c)
						line = score.readline()

					all_epochs.append(epochs)
					all_probs.append(probs)
					all_corrects.append(corrects)
					
					legend.append(size_l2+","+size_l1+","+size_in+","+act+",drop:"+dropout+",lr:"+lr) 


	if len(all_epochs) > 0:
		net_name = str(snakemake.params["show_size_l2"])+str(snakemake.params["show_size_l1"])+str(snakemake.params["show_size_in"])+str(snakemake.params["show_act"])+str(snakemake.params["show_dropout"])+str(snakemake.params["show_lr"])
		colors = get_spaced_colors(len(all_epochs))	
		
		for index, probs in enumerate(all_probs):
			plt.plot(all_epochs[index], probs, color=colors[index])
		plt.xticks([x*64 for x in range(int(all_epochs[0][-1]/64))])
		save("probs", size_to_show, net_name)
		
		for index, corrects in enumerate(all_corrects):
			plt.plot(all_epochs[index], corrects, color=colors[index])
		plt.xticks([x*64 for x in range(int(all_epochs[0][-1]/64))])
		save("corrects", size_to_show, net_name)

		print("DONE", net_name)

	else:
		print("FAILED")



type_c = len(snakemake.params["types"])

for in_score in score_files:
	if in_score[:6] == "SCORES":
		#	[SCORE, 200, 2048, 1024, 64, tanh, drop, 0, lr, 0.001, batch, 256]
		_, size_out, size_l2, size_l1, size_in, act, _, dropout, _, lr = in_score[:-4].split("_")

		if size_out == size_to_show \
			and (len(snakemake.params["show_size_in"]) == 0 or size_in  in snakemake.params["show_size_in"]) \
			and (len(snakemake.params["show_size_l2"]) == 0 or size_l2  in snakemake.params["show_size_l2"]) \
			and (len(snakemake.params["show_size_l1"]) == 0 or size_l1  in snakemake.params["show_size_l1"]) \
			and (len(snakemake.params["show_act"]) 	   == 0 or act 		in snakemake.params["show_act"]) \
			and (len(snakemake.params["show_dropout"]) == 0 or dropout  in snakemake.params["show_dropout"]) \
			and (len(snakemake.params["show_lr"]) 	   == 0 or lr 		in snakemake.params["show_lr"]) :

			epoch = 0
			with open(score_dir+in_score, "r") as score:
				line = score.readline()
				while line and epoch < 640:
					items = line.split("#")
					delta_epoch = int(items[1])
					epoch += delta_epoch
					line_probs = items[2][6:].split(" ")[:-1]
					line_corrects = items[3][9:].split(" ")[:-1]

					val_rounds = int(len(line_probs)/type_c)
					
					avg_probs = [sum([float(line_probs[typ+val*type_c]) for val in range(val_rounds)])/val_rounds for typ in range(type_c)]
					min_corrects = [decide_correct([line_corrects[typ+val*type_c] for val in range(val_rounds)]) for typ in range(type_c)]

					ys = [float(prob)*100 for prob in avg_probs]
					xs = [epoch for prob in avg_probs]
					plt.scatter(x=xs, y=ys, alpha=0.6, \
						 		color=['green' if correct == "1" else 'yellow' if correct == "0.5" else 'red' for correct in min_corrects])

					line = score.readline()

			plt.ylim(0, 105)
			plt.yticks([0, 25, 50, 75, 100])
			plt.xticks([x*64 for x in range(int(epoch/64)+1)])
			plt.grid()
			net_name = in_score[7:].replace(".", "")
			plt.savefig(snakemake.params["path_to_figures"]+"DIST_"+net_name+".png")
			plt.clf()
			print("DONE DIST", net_name)
