configfile: "config.yaml" 	

rule all: 
	input:
		"score.txt"

rule all_types_to_fa: 
	input: 
		path=config["path_all_types"]
	output: 
		fa="data/ref/t{type}.fa"
	script:
		"scripts/allTypesToFasta.py"

rule simulate_train:
	input:
		"data/ref/t{type}.fa"
	output: 
		"data/reads/r{type}.t.{training_round}.fq",
		temp("data/reads/trash{type}.t.{training_round}.fq")
	params:
		read_length=config["wgsim_read_length"],
		distance=config["wgsim_distance"],
		error=config["wgsim_error"],
		read_count=config["wgsim_read_count"]	
	conda:
		"envs/main_env.yaml"
	shell:
		"wgsim -1 {params.read_length} -2 {params.read_length} -d {params.distance} -e {params.error} -N {params.read_count} -h {input} {output} &>/dev/null"

rule simulate_val:
	input:
		"data/ref/t{type}.fa"
	output:
		"data/reads/r{type}.v.{validation_round}.fq",
		temp("data/reads/trash{type}.{validation_round}.v.fq")
	params:
		read_length=config["wgsim_read_length"],
		distance=config["wgsim_distance"],
		error=config["wgsim_error"], 
		read_count=config["wgsim_read_count"]	
	conda: 
		"envs/main_env.yaml" 
	shell:
		"wgsim -1 {params.read_length} -2 {params.read_length} -d {params.distance} -e {params.error} -N {params.read_count} -h {input} {output} &>/dev/null"

rule train_net:
	input:
		train_reads=expand("data/reads/r{type}.t.{training_round}.fq", type=config["types"], training_round=range(config["training_rounds"])),
		val_reads=expand("data/reads/r{type}.v.{validation_round}.fq", type=config["types"], validation_round=range(config["validation_rounds"]))
	output:
		scores=temp("score.txt")
	params: 
		types=config["types"],
		training_rounds=config["training_rounds"],
		validation_rounds=config["validation_rounds"],
		batch_size=config["batch_size"],
		ks=config["ks"],
		size_l1s=config["size_l1s"],
		size_l2s=config["size_l2s"],
		activation_functions=config["activation_functions"],
		dropout_probs=config["dropout_probs"],
		learning_rates=config["learning_rates"]
	conda:
		"envs/main_env.yaml"
	script:
		"scripts/trainTestNN.py"

rule view_scores:
	params:
		types=config["types"],
		show_size_out=config["show_size_out"],
		show_size_l2=config["show_size_l2"],
		show_size_l1=config["show_size_l1"],
		show_size_in=config["show_size_in"],
		show_act=config["show_act"],
		show_dropout=config["show_dropout"],
		show_lr=config["show_lr"],
		path_to_figures=config["path_to_figures"]
	conda:
		"envs/main_env.yaml"
	script:
		"scripts/scoreViewer.py"
