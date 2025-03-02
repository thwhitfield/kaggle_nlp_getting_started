# Standard commands for setting up and running experiments

save-environment:
	conda env export --no-builds > setup/environment.yml

load-environment:
	conda env create -f setup/environment.yml

# kaggle
download-data:
	kaggle competitions download -c nlp-getting-started -p data/
	unzip data/nlp-getting-started.zip -d data/

# Start mlflow and prefect servers
start-services:
	bash scripts/start_services.sh

# Shut down mlflow and prefect servers
stop-services:
	bash scripts/stop_services.sh

# Stop any running mlflow servers
stop-mlflow-servers:
	pkill -f gunicorn
