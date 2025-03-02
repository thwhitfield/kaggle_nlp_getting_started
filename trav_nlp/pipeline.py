import datetime
import logging
import os
import sys
import warnings
import zipfile
from pathlib import Path

import kaggle
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import polars as pl
from gensim import downloader
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from prefect import flow, task
from prefect.cache_policies import DEFAULT, INPUTS
from prefect.context import get_run_context  # New import
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from trav_nlp.misc import (
    flatten_dict,
    get_git_commit_hash,
    polars_train_test_split,
    polars_train_val_test_split,
    submit_to_kaggle,
    verify_git_commit,
)

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
def train_val_test_split(
    data_file: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_seed: int = None,
):
    """
    Create train, validation, and test splits using Polars.
    """
    # Load dataset with Polars
    df = pl.read_csv(data_file)

    # Split data using polars_train_test_split
    df_train, df_val, df_test = polars_train_val_test_split(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        shuffle=True,
        seed=random_seed,
    )

    return df_train, df_val, df_test


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that converts a collection of texts to their embedding representations.
    The embeddings for each text are computed by averaging the word embeddings.
    """

    def __init__(self, embeddings, aggregation="mean"):
        self.embeddings = embeddings
        self.aggregation = aggregation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = []
        for text in X:
            words = (
                text.split()
            )  # simple whitespace tokenizer; replace with a better tokenizer if needed
            # Collect embeddings for words that exist in our vocabulary
            word_embs = [
                self.embeddings[word] for word in words if word in self.embeddings
            ]
            if word_embs:
                if self.aggregation == "mean":
                    emb = np.mean(word_embs, axis=0)
                elif self.aggregation == "sum":
                    emb = np.sum(word_embs, axis=0)
                else:
                    raise ValueError(
                        "Unsupported aggregation method: choose 'mean' or 'sum'"
                    )
            else:
                # If none of the words are in our embeddings, return a zero vector
                # emb = np.zeros(len(next(iter(self.embeddings.values()))))
                emb = np.zeros(len(next(iter(self.embeddings))))
            transformed_X.append(emb)
        return np.array(transformed_X)


@task
def train(df_train, df_val=None, full_train=False, model_params={}, embeddings=None):
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

    if embeddings is None:
        # Create the pipeline with the text selector, vectorizer, and classifier
        pipeline = make_pipeline(
            extract_text_transform,
            CountVectorizer(),
            convert_to_numpy_transform,
            lgb.LGBMClassifier(**model_params),
        )
    else:
        pipeline = make_pipeline(
            extract_text_transform,
            EmbeddingVectorizer(embeddings),
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


@task
def eval_df_test(pipeline, df_test):
    """
    Evaluate the model on the test dataset.

    Args:
        pipeline (Pipeline): Trained model pipeline.
        df_test (DataFrame): Test dataset.

    Returns:
        None
    """
    test_preds = pipeline.predict_proba(df_test)[:, 1]
    test_roc_auc = roc_auc_score(df_test["target"], test_preds)
    logging.info(f"Test ROC: {test_roc_auc}")
    mlflow.log_metric("test_roc_auc", test_roc_auc)


@task
def generate_and_submit_to_kaggle(
    pipeline, kaggle_test_path, kaggle_sample_submission_path, submissions_dir
):
    """
    Generate predictions and submit to Kaggle.

    Args:
        pipeline (Pipeline): Trained model pipeline.
        kaggle_test_path (str): Path to Kaggle test dataset.
        kaggle_sample_submission_path (str): Path to Kaggle sample submission file.

    Returns:
        None
    """
    df_kaggle_test = pl.read_csv(kaggle_test_path)
    kaggle_sample_submission = pl.read_csv(kaggle_sample_submission_path)

    kaggle_test_preds = pipeline.predict(df_kaggle_test)
    kaggle_sample_submission = kaggle_sample_submission.with_columns(
        pl.Series("target", kaggle_test_preds)
    )

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    filename = f"submission_{timestamp}.csv"
    submission_path = Path(submissions_dir) / filename

    kaggle_sample_submission.write_csv(submission_path)

    kaggle_score = submit_to_kaggle("nlp-getting-started", submission_path)
    logging.info(f"Public Kaggle score: {kaggle_score}")
    mlflow.log_metric("public_kaggle_score", kaggle_score)


def suggest_parameters(trial, param_ranges):
    """
    Suggest parameters using Optuna trial.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        param_ranges (dict): Parameter ranges for tuning.

    Returns:
        dict: Suggested parameters.
    """
    tuning_params = {}
    for param, spec in param_ranges.items():
        if spec["type"] == "int":
            tuning_params[param] = trial.suggest_int(param, spec["low"], spec["high"])
        elif spec["type"] == "float":
            if spec.get("log", False):
                tuning_params[param] = trial.suggest_float(
                    param, spec["low"], spec["high"], log=True
                )
            else:
                tuning_params[param] = trial.suggest_float(
                    param, spec["low"], spec["high"]
                )
        elif spec["type"] == "categorical":
            tuning_params[param] = trial.suggest_categorical(param, spec["choices"])
    return tuning_params


@task
def tune_hyperparameters(
    df_train, df_val, model_params_base, param_ranges, n_trials=10, embeddings=None
):
    """
    Tune hyperparameters using Optuna.

    Args:
        df_train (DataFrame): Training dataset.
        df_val (DataFrame): Validation dataset.
        model_params_base (dict): Base model parameters.
        param_ranges (dict): Parameter ranges for tuning.
        n_trials (int, optional): Number of tuning trials. Defaults to 10.

    Returns:
        tuple: Best model, best parameters, and best metric value.
    """

    def objective(trial):

        with mlflow.start_run(nested=True):
            tuning_params = model_params_base.copy()
            # Log parameters to MLflow with prefix
            mlflow_params = {}
            suggested_params = suggest_parameters(trial, param_ranges)
            tuning_params.update(suggested_params)

            # Store parameter with prefix for MLflow logging
            mlflow_params = {
                f"model_params.{param}": value
                for param, value in suggested_params.items()
            }

            # Log the trial's parameters to MLflow
            mlflow.log_params(mlflow_params)

            model = train(
                df_train,
                df_val,
                full_train=False,
                model_params=tuning_params,
                embeddings=None,
            )
            val_preds = model.predict_proba(df_val)[:, 1]
        return roc_auc_score(df_val["target"], val_preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_trial_params = study.best_trial.params
    best_params = model_params_base.copy()
    best_params.update(best_trial_params)
    best_model = train(df_train, df_val, full_train=False, model_params=best_params)
    return best_model, best_params, study.best_value


@flow
def run_pipeline(cfg: DictConfig):
    ctx = get_run_context()
    run_name = getattr(ctx.flow_run, "name", "default_run")  # Retrieve Prefect run name
    with mlflow.start_run(run_name=run_name) as run:

        # If the cfg.git_commit_hash is populated, that should mean that we're re-running a previous
        # pipeline run. In that case we check to make sure that we're on the same git commit,
        # otherwise we'll raise a RuntimeError
        if cfg.git_commit_hash is not None:
            curr_commit_hash = verify_git_commit(CURRENT_DIR)
            if cfg.git_commit_hash != curr_commit_hash:
                raise RuntimeError(
                    f"Commit hash in cfg file, {cfg.git_commit_hash}, not equal to current cfg hash, {curr_commit_hash}"
                )
        elif not cfg.test_run:
            # Verify that the trav_nlp folder doesn't have any uncommitted changes to it
            git_commit_hash = verify_git_commit(CURRENT_DIR)
            cfg.git_commit_hash = git_commit_hash

        # Capture MLflow run id and store in config
        cfg.mlflow.run_id = run.info.run_id
        # Save the used config under a filename based on the run_id.
        config_save_dir = "/Users/traviswhitfield/Documents/github/kaggle_nlp_getting_started/config_history"
        os.makedirs(config_save_dir, exist_ok=True)
        config_save_path = os.path.join(config_save_dir, f"{run_name}.yaml")
        OmegaConf.save(cfg, config_save_path, resolve=True)
        mlflow.log_artifact(config_save_path)

        mlflow.set_tags(OmegaConf.to_container(cfg.mlflow.tags, resolve=True))

        if cfg.embeddings.name == "gensim":
            embeddings = downloader.load(cfg.embeddings.gensim_embedding_name)
        elif cfg.embeddings.name == "count_vectorizer":
            embeddings = None

        # Call download_kaggle_data using config parameters
        download_kaggle_data(
            data_dir=cfg.data.raw_dir,
            competition_name=cfg.kaggle_competition,
        )
        # Call train_test_split using specified parameters from config
        df_train, df_val, df_test = train_val_test_split(
            data_file=cfg.data.raw_train_path,
            train_frac=cfg.split_params.train_frac,
            val_frac=cfg.split_params.val_frac,
            test_frac=cfg.split_params.test_frac,
            random_seed=cfg.split_params.train_val_test_seed,
        )

        if cfg.run_hyperparameter_tuning:
            # Hyperparameter tuning is enabled.
            # Pass the new hyperparameter parameters structure from config to the tuner.
            best_model, best_params, best_metric = tune_hyperparameters(
                df_train,
                df_val,
                cfg.model_params,
                OmegaConf.to_container(cfg.hyperparameter_tuning.parameters),
                n_trials=cfg.hyperparameter_tuning.n_trials,
                embeddings=embeddings,
            )
            # for key, value in best_params.items():
            #     mlflow.log_param(f"best_model_params.{key}", value)
            # mlflow.log_metric("best_val_roc_auc", best_metric)
            # Update the config with the best parameters and re-save
            cfg.model_params.update(best_params)
            OmegaConf.save(cfg, config_save_path, resolve=True)
            mlflow.log_artifact(config_save_path)
            model = best_model
        else:
            model = train(
                df_train, df_val, model_params=cfg.model_params, embeddings=embeddings
            )

        eval_df_test(model, df_test)

        # Removed automatic Kaggle submission.
        logging.info(
            "Pipeline run complete. Review test set performance before manual submission using submit_pipeline_run()."
        )

        # Log the parameters at the end so that the cfg can be updated first (if needed)
        mlflow.log_params(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))


# Updated submission flow: remove internal mlflow.run so that external CLI can resume the run.
@flow(name="submit_pipeline_run")
def submit_pipeline_run(run_identifier: str, cfg: DictConfig):
    # No mlflow.start_run wrapper here because the CLI wraps this flow.
    df_train, df_val, df_test = train_val_test_split(
        data_file=cfg.data.raw_train_path,
        train_frac=cfg.split_params.train_frac,
        val_frac=cfg.split_params.val_frac,
        test_frac=cfg.split_params.test_frac,
        random_seed=cfg.split_params.train_val_test_seed,
    )
    df_full_train = pl.concat([df_train, df_val, df_test])
    full_pipeline = train(df_full_train, model_params=cfg.model_params, full_train=True)
    generate_and_submit_to_kaggle(
        full_pipeline,
        cfg.data.raw_test_path,
        cfg.data.raw_sample_submission_path,
        submissions_dir=cfg.data.submissions_dir,
    )
    logging.info(f"Submission for run '{run_identifier}' completed.")


if __name__ == "__main__":
    with initialize(config_path="../conf", job_name="run_pipeline", version_base=None):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
        run_pipeline(cfg)
