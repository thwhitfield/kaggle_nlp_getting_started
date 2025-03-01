import logging
import os
import sys
import warnings
import zipfile
from pathlib import Path

import kaggle
import lightgbm as lgb
import mlflow
import polars as pl

# Remove hydra.main and import initialize/compose instead
from hydra import compose, initialize
from omegaconf import DictConfig
from prefect import flow, task
from prefect.cache_policies import DEFAULT, INPUTS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from trav_nlp.misc import polars_train_test_split, verify_git_commit

# Get the directory containing the current file
CURRENT_DIR = Path(__file__).parent

# Filter all the LGBMClassifier not valid feature names warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


@task(cache_policy=INPUTS)
def download_kaggle_data(
    data_dir: str = "data", competition_name: str = "nlp-getting-started"
):
    """
    Downloads data from a Kaggle competition and unzips it.
    """
    try:
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Download the competition data using the Kaggle API
        kaggle.api.competition_download_cli(competition=competition_name, path=data_dir)

        # Get the path to the downloaded zip file
        zip_file_path = os.path.join(data_dir, f"{competition_name}.zip")

        # Unzip the downloaded files
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        # Remove the zip file
        os.remove(zip_file_path)

        print(f"Data downloaded and extracted to {data_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")


@task(cache_policy=INPUTS)
def train_test_split(
    data_file: str,
    train_frac: float,
    test_frac: float,
    random_seed: int = None,
):
    """
    Create train, validation, and test splits using Polars.
    """
    # Load dataset with Polars
    df = pl.read_csv(data_file)

    # Split data using polars_train_test_split
    df_train, df_test = polars_train_test_split(
        df,
        train_frac=train_frac,
        test_frac=test_frac,
        shuffle=True,
        seed=random_seed,
    )

    return df_train, df_test


@task
def train(df_train, df_val=None, full_train=False, model_params={}):
    """
    Train and optimize the model.

    Args:
        df_train (DataFrame): Training dataset.
        df_val (DataFrame, optional): Validation dataset. Defaults to None.
        full_train (bool, optional): Whether to train on the full dataset. Defaults to False.
        model_params (dict, optional): Parameters for the model. Defaults to {}.

    Returns:
        Pipeline: Trained model pipeline.
    """

    # Define a function to extract the 'text' column
    def extract_text(df):
        return df["text"]

    def convert_to_numpy(scipy_csr_matrix):
        return scipy_csr_matrix.toarray()

    # Create a FunctionTransformer to apply that function
    extract_text_transform = FunctionTransformer(extract_text)

    convert_to_numpy_transform = FunctionTransformer(convert_to_numpy)

    # Create the pipeline with the text selector, vectorizer, and classifier
    pipeline = make_pipeline(
        extract_text_transform,
        CountVectorizer(),
        convert_to_numpy_transform,
        lgb.LGBMClassifier(**model_params),
    )

    pipeline.fit(df_train, df_train["target"])

    train_preds = pipeline.predict_proba(df_train)[:, 1]
    train_roc_auc = roc_auc_score(df_train["target"], train_preds)

    if not full_train:
        logging.info(f"Train ROC: {train_roc_auc}")
        mlflow.log_metric("train_roc_auc", train_roc_auc)

    if df_val is not None:
        val_preds = pipeline.predict_proba(df_val)[:, 1]
        val_roc_auc = roc_auc_score(df_val["target"], val_preds)
        logging.info(f"Val ROC: {val_roc_auc}")
        mlflow.log_metric("val_roc_auc", val_roc_auc)

    return pipeline


def eval_on_test():
    pass


def submit_to_kaggle():
    pass


@flow
def run_pipeline(cfg: DictConfig):

    # Verify that the trav_nlp folder doesn't have any uncommitted changes to it
    # commit_hash = verify_git_commit(CURRENT_DIR)

    # Call download_kaggle_data using config parameters
    download_kaggle_data(
        data_dir=cfg.kaggle.data_dir,
        competition_name=cfg.kaggle.competition,
    )
    # Call train_test_split using specified parameters from config
    df_train, df_test = train_test_split(
        data_file=cfg.kaggle.train_file,
        train_frac=cfg.params.train_frac,
        test_frac=cfg.params.test_frac,
        random_seed=cfg.params.train_val_test_seed,
    )

    # Train the model using the training data
    model = train(
        df_train,
        df_val=None,
        full_train=False,
        model_params=cfg.model_params,
    )


if __name__ == "__main__":
    with initialize(config_path="../conf", job_name="run_pipeline", version_base=None):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
        run_pipeline(cfg)
