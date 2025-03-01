import os
import sys

import mlflow
from omegaconf import OmegaConf
from trav_nlp.pipeline import submit_pipeline_run


def main():
    if len(sys.argv) != 2:
        print("Usage: python submit_pipeline_run.py <mlflow_run_id>")
        sys.exit(1)
    run_id = sys.argv[1]
    config_save_path = f"/Users/traviswhitfield/Documents/github/kaggle_nlp_getting_started/config_history/{run_id}.yaml"
    if not os.path.exists(config_save_path):
        print(f"Config for run {run_id} not found: {config_save_path}")
        sys.exit(1)
    cfg = OmegaConf.load(config_save_path)
    # Resume the original MLflow run so that logs are appended to it.
    with mlflow.start_run(run_id=cfg.mlflow.run_id, nested=True):
        submit_pipeline_run(run_id, cfg)


if __name__ == "__main__":
    main()
