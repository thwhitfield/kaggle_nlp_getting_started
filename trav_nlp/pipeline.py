import argparse
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
from prefect import flow, get_run_logger, task
from prefect.cache_policies import DEFAULT, INPUTS
from prefect.context import get_run_context
from prefect.settings import PREFECT_LOGGING_LEVEL
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
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
ROOT_DIR = CURRENT_DIR.parent

# Filter all the LGBMClassifier not valid feature names warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

# Filter Pipeline not fitted yet warnings
warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet",
    category=FutureWarning,
)


# Example usage
print(f"Current Prefect logging level: {PREFECT_LOGGING_LEVEL.value()}")


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


def cross_validate_model(
    X, y, pipeline_constructor, cv=5, stratified=True, random_seed=42
):
    """
    Perform cross-validation on a model.

    Args:
        X: Features data
        y: Target labels
        pipeline_constructor: Function to create a new pipeline
        cv (int): Number of folds for cross-validation
        stratified (bool): Whether to use stratified folds
        random_seed (int): Random seed for reproducibility

    Returns:
        tuple: Mean scores and list of trained models
    """
    logger = get_run_logger()

    if stratified:
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_seed)
    else:
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed)

    scores = {"f1": [], "roc_auc": []}
    models = []

    for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create fresh pipeline for each fold
        pipeline = pipeline_constructor()
        pipeline.fit(X_train, y_train)

        # Evaluate on validation fold
        val_preds_proba = pipeline.predict_proba(X_val)[:, 1]
        val_preds = pipeline.predict(X_val)
        fold_roc_auc = roc_auc_score(y_val, val_preds_proba)
        fold_f1 = f1_score(y_val, val_preds)

        logger.info(f"Fold {i+1}/{cv} - ROC AUC: {fold_roc_auc:.4f}, F1: {fold_f1:.4f}")

        scores["f1"].append(fold_f1)
        scores["roc_auc"].append(fold_roc_auc)
        models.append(pipeline)

    mean_f1 = np.mean(scores["f1"])
    mean_roc_auc = np.mean(scores["roc_auc"])
    std_f1 = np.std(scores["f1"])
    std_roc_auc = np.std(scores["roc_auc"])

    logger.info(f"CV Mean ROC AUC: {mean_roc_auc:.4f} (±{std_roc_auc:.4f})")
    logger.info(f"CV Mean F1: {mean_f1:.4f} (±{std_f1:.4f})")

    # Log metrics to MLflow
    mlflow.log_metric("cv_roc_auc_mean", mean_roc_auc)
    mlflow.log_metric("cv_roc_auc_std", std_roc_auc)
    mlflow.log_metric("cv_f1_mean", mean_f1)
    mlflow.log_metric("cv_f1_std", std_f1)

    return {
        "mean_roc_auc": mean_roc_auc,
        "mean_f1": mean_f1,
        "std_roc_auc": std_roc_auc,
        "std_f1": std_f1,
    }, models


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that converts a collection of texts to their embedding representations.
    The embeddings for each text are computed by applying one or more aggregation methods
    to the word embeddings (mean, sum, min, max).

    Parameters:
    -----------
    embeddings : dict or similar
        Word embeddings dictionary where keys are words and values are embeddings
    aggregation : str or list, default=["mean"]
        Aggregation method(s) to use. Can be a single string or a list of strings.
        Supported methods: "mean", "sum", "min", "max"
    """

    def __init__(self, embeddings, aggregation=["mean"]):
        self.embeddings = embeddings
        # Convert string to list if a single aggregation method is provided
        if isinstance(aggregation, str):
            self.aggregation = [aggregation]
        else:
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

            if not word_embs:
                # If none of the words are in our embeddings, return a zero vector for each aggregation method
                emb_dim = len(next(iter(self.embeddings)))
                emb = np.zeros(emb_dim * len(self.aggregation))
            else:
                # Apply each aggregation method and concatenate the results
                aggregated_embs = []
                for agg_method in self.aggregation:
                    if agg_method == "mean":
                        agg_emb = np.mean(word_embs, axis=0)
                    elif agg_method == "sum":
                        agg_emb = np.sum(word_embs, axis=0)
                    elif agg_method == "min":
                        agg_emb = np.min(word_embs, axis=0)
                    elif agg_method == "max":
                        agg_emb = np.max(word_embs, axis=0)
                    else:
                        raise ValueError(
                            "Unsupported aggregation method: choose from 'mean', 'sum', 'min', or 'max'"
                        )
                    aggregated_embs.append(agg_emb)

                # Concatenate all aggregation results
                emb = np.concatenate(aggregated_embs)

            transformed_X.append(emb)
        return np.array(transformed_X)


class OptimizedThresholdClassifier:
    """
    A wrapper for classifiers that optimizes the decision threshold to maximize F1 score.
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self.threshold = 0.5  # Default threshold

    def fit(self, X, y):
        """
        Fits the underlying classifier and optimizes the decision threshold
        using the precision-recall curve to maximize F1 score.
        """
        self.classifier = self.classifier.fit(X, y)
        y_scores = self.classifier.predict_proba(X)[:, 1]

        # Calculate precision and recall values for different thresholds
        precision, recall, thresholds = precision_recall_curve(y, y_scores)

        # Calculate F1 score for each threshold
        # Note: precision_recall_curve returns one more precision/recall value than thresholds
        f1_scores = [2 * (p * r) / (p + r + 1e-10) for p, r in zip(precision, recall)]

        # Find the threshold that gives the best F1 score
        # -1 because precision_recall_curve returns one more precision/recall value
        if len(thresholds) > 0:
            best_idx = np.argmax(f1_scores[:-1])
            self.threshold = thresholds[best_idx]

        return self

    def predict(self, X):
        """
        Predicts classes using the optimized threshold.
        """
        y_scores = self.classifier.predict_proba(X)[:, 1]
        return (y_scores >= self.threshold).astype(int)

    def predict_proba(self, X):
        """
        Returns probability estimates.
        """
        return self.classifier.predict_proba(X)


@task
def train(
    df_train,
    df_val=None,
    full_train=False,
    model_params=None,
    embeddings=None,
    embedding_aggregations=None,
    use_cross_validation=False,
    cv_folds=5,
    stratified_cv=True,
    cv_random_seed=42,
):
    """
    Train and optimize the model.

    Args:
        df_train (DataFrame): Training dataset.
        df_val (DataFrame, optional): Validation dataset. Defaults to None.
        full_train (bool, optional): Whether to train on the full dataset. Defaults to False.
        model_params (dict, optional): Parameters for the model. Defaults to {}.
        embeddings (dict, optional): Word embeddings dictionary. Defaults to None.
        embedding_aggregations (list, optional): Aggregation methods for embeddings. Defaults to None.
        use_cross_validation (bool, optional): Whether to use cross-validation. Defaults to False.
        cv_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        stratified_cv (bool, optional): Whether to use stratified folds. Defaults to True.
        cv_random_seed (int, optional): Random seed for cross-validation. Defaults to 42.

    Returns:
        Pipeline: Trained model pipeline.
    """
    logger = get_run_logger()  # Use Prefect logger

    if model_params is None:
        model_params = {}

    # If cross-validation is enabled and we have df_val, combine with df_train
    if use_cross_validation and df_val is not None and not full_train:
        logger.info("Using cross-validation - combining train and validation sets")
        df_train = pl.concat([df_train, df_val])

    # Define a function to create a fresh pipeline
    def create_pipeline():
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
            # Use the OptimizedThresholdClassifier to wrap LGBMClassifier
            return make_pipeline(
                extract_text_transform,
                CountVectorizer(),
                convert_to_numpy_transform,
                OptimizedThresholdClassifier(lgb.LGBMClassifier(**model_params)),
            )
        else:
            return make_pipeline(
                extract_text_transform,
                EmbeddingVectorizer(embeddings, embedding_aggregations),
                OptimizedThresholdClassifier(lgb.LGBMClassifier(**model_params)),
            )

    # If using cross-validation and not doing full training
    if use_cross_validation and not full_train:
        # Preparing data for cross-validation
        X = df_train.to_pandas()  # Convert to pandas for sklearn compatibility
        y = X["target"]

        cv_results, cv_models = cross_validate_model(
            X,
            y,
            create_pipeline,
            cv=cv_folds,
            stratified=stratified_cv,
            random_seed=cv_random_seed,
        )

        # Fit final model on full training data
        pipeline = create_pipeline()
        pipeline = pipeline.fit(df_train, df_train["target"])

        # Log the optimized threshold
        optimized_threshold = pipeline.steps[-1][1].threshold
        logger.info(f"Final model optimized threshold: {optimized_threshold}")
        mlflow.log_param("optimized_threshold", optimized_threshold)

        return pipeline

    # Original training process without cross-validation
    pipeline = create_pipeline()
    pipeline = pipeline.fit(df_train, df_train["target"])

    # Log the optimized threshold
    optimized_threshold = pipeline.steps[-1][1].threshold
    logger.info(f"Optimized threshold: {optimized_threshold}")
    mlflow.log_param("optimized_threshold", optimized_threshold)

    # Calculate ROC AUC and F1 scores for training data
    train_preds_proba = pipeline.predict_proba(df_train)[:, 1]
    train_preds = pipeline.predict(df_train)
    train_roc_auc = roc_auc_score(df_train["target"], train_preds_proba)
    train_f1 = f1_score(df_train["target"], train_preds)

    if not full_train:
        logger.info(f"Train ROC: {train_roc_auc}, Train F1: {train_f1}")
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        mlflow.log_metric("train_f1", train_f1)

    if df_val is not None and not use_cross_validation:
        val_preds_proba = pipeline.predict_proba(df_val)[:, 1]
        val_preds = pipeline.predict(df_val)
        val_roc_auc = roc_auc_score(df_val["target"], val_preds_proba)
        val_f1 = f1_score(df_val["target"], val_preds)
        logger.info(f"Val ROC: {val_roc_auc}, Val F1: {val_f1}")
        mlflow.log_metric("val_roc_auc", val_roc_auc)
        mlflow.log_metric("val_f1", val_f1)

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
    logger = get_run_logger()  # Use Prefect logger
    test_preds_proba = pipeline.predict_proba(df_test)[:, 1]
    test_preds = pipeline.predict(df_test)
    test_roc_auc = roc_auc_score(df_test["target"], test_preds_proba)
    test_f1 = f1_score(df_test["target"], test_preds)
    logger.info(f"Test ROC: {test_roc_auc}, Test F1: {test_f1}")
    mlflow.log_metric("test_roc_auc", test_roc_auc)
    mlflow.log_metric("test_f1", test_f1)


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
    df_train,
    df_val,
    model_params_base,
    param_ranges,
    n_trials=10,
    embeddings=None,
    embedding_aggregations=None,
    use_cross_validation=False,
    cv_folds=5,
    stratified_cv=True,
    cv_random_seed=42,
):
    """
    Tune hyperparameters using Optuna.

    Args:
        df_train (DataFrame): Training dataset.
        df_val (DataFrame): Validation dataset.
        model_params_base (dict): Base model parameters.
        param_ranges (dict): Parameter ranges for tuning.
        n_trials (int, optional): Number of tuning trials. Defaults to 10.
        embeddings (dict, optional): Word embeddings dictionary. Defaults to None.
        embedding_aggregations (list, optional): Aggregation methods for embeddings. Defaults to None.
        use_cross_validation (bool, optional): Whether to use cross-validation. Defaults to False.
        cv_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        stratified_cv (bool, optional): Whether to use stratified folds. Defaults to True.
        cv_random_seed (int, optional): Random seed for cross-validation. Defaults to 42.

    Returns:
        tuple: Best model, best parameters, and best metric value.
    """

    # If cross-validation is enabled, combine train and validation sets
    if use_cross_validation:
        logger = get_run_logger()  # Use Prefect logger
        logger.info(
            "Using cross-validation for hyperparameter tuning - combining train and validation sets"
        )
        df_combined = pl.concat([df_train, df_val])
    else:
        df_combined = df_train

    def objective(trial):
        with mlflow.start_run(nested=True):
            logger = get_run_logger()  # Use Prefect logger
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

            if use_cross_validation:
                model = train(
                    df_combined,
                    full_train=False,
                    model_params=tuning_params,
                    embeddings=embeddings,
                    embedding_aggregations=embedding_aggregations,
                    use_cross_validation=True,
                    cv_folds=cv_folds,
                    stratified_cv=stratified_cv,
                    cv_random_seed=cv_random_seed,
                )
                # Get the CV metric from the MLflow run
                return mlflow.get_run(
                    run_id=mlflow.active_run().info.run_id
                ).data.metrics["cv_roc_auc_mean"]
            else:
                model = train(
                    df_train,
                    df_val,
                    full_train=False,
                    model_params=tuning_params,
                    embeddings=embeddings,
                    embedding_aggregations=embedding_aggregations,
                )
                val_preds_proba = model.predict_proba(df_val)[:, 1]
                val_preds = model.predict(df_val)
                val_roc_auc = roc_auc_score(df_val["target"], val_preds_proba)
                val_f1 = f1_score(df_val["target"], val_preds)

                return val_roc_auc

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(OmegaConf.to_container(cfg.model_params, resolve=True))
    study.optimize(objective, n_trials=n_trials)

    best_trial_params = study.best_trial.params
    best_params = model_params_base.copy()
    best_params.update(best_trial_params)

    if use_cross_validation:
        best_model = train(
            df_combined,
            full_train=False,
            model_params=best_params,
            embeddings=embeddings,
            embedding_aggregations=embedding_aggregations,
            use_cross_validation=True,
            cv_folds=cv_folds,
            stratified_cv=stratified_cv,
            cv_random_seed=cv_random_seed,
        )
    else:
        best_model = train(
            df_train,
            df_val,
            full_train=False,
            model_params=best_params,
            embeddings=embeddings,
            embedding_aggregations=embedding_aggregations,
        )

    return best_model, best_params, study.best_value


@flow
def run_pipeline(cfg: DictConfig):
    logger = get_run_logger()  # Use Prefect logger
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
        config_save_dir = ROOT_DIR / "config_history"
        os.makedirs(config_save_dir, exist_ok=True)
        config_save_path = os.path.join(config_save_dir, f"{run_name}.yaml")
        OmegaConf.save(cfg, config_save_path, resolve=True)
        mlflow.log_artifact(config_save_path)

        mlflow.set_tags(OmegaConf.to_container(cfg.mlflow.tags, resolve=True))

        # Download the embeddings if necessary
        if cfg.embeddings.type == "gensim":
            logger.info(f"Downloading gensim embedding: {cfg.embeddings.name}")
            embeddings = downloader.load(cfg.embeddings.name)
            embedding_aggregations = cfg.embeddings.aggregations
        elif cfg.embeddings.type == "count_vectorizer":
            embeddings = None
            embedding_aggregations = None

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
                df_train=df_train,
                df_val=df_val,
                model_params_base=cfg.model_params,
                param_ranges=OmegaConf.to_container(
                    cfg.hyperparameter_tuning.parameters
                ),
                n_trials=cfg.hyperparameter_tuning.n_trials,
                embeddings=embeddings,
                embedding_aggregations=embedding_aggregations,
                use_cross_validation=cfg.get("use_cross_validation", False),
                cv_folds=cfg.get("cv_folds", 5),
                stratified_cv=cfg.get("stratified_cv", True),
                cv_random_seed=cfg.get("cv_random_seed", 42),
            )

            # Update the config with the best parameters and re-save
            cfg.model_params.update(best_params)
            OmegaConf.save(cfg, config_save_path, resolve=True)
            mlflow.log_artifact(config_save_path)
            model = best_model
        else:
            model = train(
                df_train,
                df_val,
                model_params=cfg.model_params,
                embeddings=embeddings,
                embedding_aggregations=embedding_aggregations,
                use_cross_validation=cfg.get("use_cross_validation", False),
                cv_folds=cfg.get("cv_folds", 5),
                stratified_cv=cfg.get("stratified_cv", True),
                cv_random_seed=cfg.get("cv_random_seed", 42),
            )

        eval_df_test(model, df_test)

        # Removed automatic Kaggle submission.
        logger.info(
            "Pipeline run complete. Review test set performance before manual submission using submit_pipeline_run()."
        )

        # Log the parameters at the end so that the cfg can be updated first (if needed)
        mlflow.log_params(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))


# Updated submission flow: remove internal mlflow.run so that external CLI can resume the run.
@flow(name="submit_pipeline_run")
def submit_pipeline_run(run_identifier: str, cfg: DictConfig):
    logger = get_run_logger()  # Use Prefect logger
    # No mlflow.start_run wrapper here because the CLI wraps this flow.
    df_train, df_val, df_test = train_val_test_split(
        data_file=cfg.data.raw_train_path,
        train_frac=cfg.split_params.train_frac,
        val_frac=cfg.split_params.val_frac,
        test_frac=cfg.split_params.test_frac,
        random_seed=cfg.split_params.train_val_test_seed,
    )

    # Download the embeddings if necessary
    if cfg.embeddings.type == "gensim":
        logger.info(f"Downloading gensim embedding: {cfg.embeddings.name}")
        embeddings = downloader.load(cfg.embeddings.name)
        embedding_aggregations = cfg.embeddings.aggregations
    elif cfg.embeddings.type == "count_vectorizer":
        embeddings = None
        embedding_aggregations = None

    df_full_train = pl.concat([df_train, df_val, df_test])
    full_pipeline = train(
        df_full_train,
        model_params=cfg.model_params,
        full_train=True,
        embeddings=embeddings,
        embedding_aggregations=embedding_aggregations,
    )
    generate_and_submit_to_kaggle(
        full_pipeline,
        cfg.data.raw_test_path,
        cfg.data.raw_sample_submission_path,
        submissions_dir=cfg.data.submissions_dir,
    )
    logger.info(f"Submission for run '{run_identifier}' completed.")


def load_config_from_path_or_name(config_arg):
    """
    Load configuration from either a direct path to a YAML file
    or a name of a run in the config_history folder.

    Args:
        config_arg (str): Path to a YAML file or name of a run config

    Returns:
        OmegaConf: Loaded configuration

    Raises:
        ValueError: If the config_arg does not correspond to a valid file
    """
    # Check if the provided argument is a direct path to a YAML file
    if os.path.isfile(config_arg) and (
        config_arg.endswith(".yaml") or config_arg.endswith(".yml")
    ):
        print(f"Loading config from path: {config_arg}")
        return OmegaConf.load(config_arg)

    # Check if it's a name in the config_history folder
    config_path = os.path.join(ROOT_DIR, "config_history", config_arg)
    if not config_path.endswith(".yaml"):
        config_path += ".yaml"

    if os.path.isfile(config_path):
        print(f"Loading config from run history: {config_path}")
        return OmegaConf.load(config_path)

    # If we got here, no valid config was found
    raise ValueError(
        f"Could not find configuration '{config_arg}'. "
        f"Please provide either a valid path to a YAML file or "
        f"the name of a run in the config_history folder."
    )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_config",
        type=str,
        help="Path to a YAML file or name of a previous run config in config_history folder",
    )
    # Parse known args to allow Hydra/compose to receive remaining args.
    args, remaining_args = parser.parse_known_args()

    if args.load_config:
        try:
            cfg = load_config_from_path_or_name(args.load_config)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        with initialize(
            config_path="../conf", job_name="run_pipeline", version_base=None
        ):
            cfg = compose(config_name="config", overrides=remaining_args)
    run_pipeline(cfg)
