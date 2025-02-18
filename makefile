# Standard commands for setting up and running experiments

save-environment:
	conda env export --no-builds > setup/environment.yml

load-environment:
	conda env create -f setup/environment.yml

# kaggle
download-data:
	kaggle competitions download -c nlp-getting-started -p data/
	unzip data/nlp-getting-started.zip -d data/