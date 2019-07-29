with open(snakemake.input["path"], "r") as types:
	with open(snakemake.output["fa"], "w+") as out:
		line = types.readline()[:-1]
		while line :
			t, seq = line.split(",")
			if t == "t" + snakemake.wildcards["type"]:
				out.write(">" + t + "\n" + seq + "\n")
				break
			line = types.readline()[:-1]

