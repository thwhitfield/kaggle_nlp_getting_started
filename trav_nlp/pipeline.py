import datetime
import logging
import warnings
from pathlib import Path

import hydra
import lightgbm as lgb
import mlflow
import optuna
import polars as pl
from omegaconf import OmegaConf
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from trav_nlp.misc import flatten_dict, polars_train_val_test_split, submit_to_kaggle

# Filter all the LGBMClassifier not valid feature names warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


def load_or_create_data(cfg):

    # Define local variables for configuration paths
    train_path = cfg.training_data.train_path
    val_path = cfg.training_data.val_path
    test_path = cfg.training_data.test_path
    raw_train_path = cfg.raw_data.train_path

    # Define local variables for parameters
    train_frac = cfg.params.train_frac
    val_frac = cfg.params.val_frac
    test_frac = cfg.params.test_frac
    seed = cfg.params.train_val_test_seed

    if Path(train_path).exists():
        df_train = pl.read_parquet(train_path)
        df_val = pl.read_parquet(val_path)
        df_test = pl.read_parquet(test_path)
    else:
        df = pl.read_csv(raw_train_path)
        Path(train_path).parent.mkdir(parents=True, exist_ok=True)

        df_train, df_val, df_test = polars_train_val_test_split(
            df,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            shuffle=True,
            seed=seed,
        )

        df_train.write_parquet(train_path)
        df_val.write_parquet(val_path)
        df_test.write_parquet(test_path)

    return df_train, df_val, df_test


def train(df_train, df_val=None, full_train=False, model_params={}):
    """Train and optimize the model"""

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


def tune_hyperparameters(
    df_train, df_val, model_params_base, param_ranges, n_trials=10
):
    def objective(trial):

        with mlflow.start_run(nested=True):
            tuning_params = model_params_base.copy()
            # Log parameters to MLflow with prefix
            mlflow_params = {}
            for param, spec in param_ranges.items():
                if spec["type"] == "int":
                    tuning_params[param] = trial.suggest_int(
                        param, spec["low"], spec["high"]
                    )
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
                    tuning_params[param] = trial.suggest_categorical(
                        param, spec["choices"]
                    )

                # Store parameter with prefix for MLflow logging
                mlflow_params[f"model_params.{param}"] = tuning_params[param]

            # Log the trial's parameters to MLflow
            mlflow.log_params(mlflow_params)

            model = train(
                df_train, df_val, full_train=False, model_params=tuning_params
            )
            val_preds = model.predict_proba(df_val)[:, 1]
        return roc_auc_score(df_val["target"], val_preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_trial_params = study.best_trial.params
    best_params = model_params_base.copy()
    best_params.update(best_trial_params)
    best_model = train(df_train, df_val, full_train=False, model_params=best_params)
    return best_model, best_params, study.best_value


def eval_df_test(pipeline, df_test):

    test_preds = pipeline.predict_proba(df_test)[:, 1]
    test_roc_auc = roc_auc_score(df_test["target"], test_preds)
    logging.info(f"Test ROC: {test_roc_auc}")
    mlflow.log_metric("test_roc_auc", test_roc_auc)


def generate_and_submit_to_kaggle(
    pipeline, kaggle_test_path, kaggle_sample_submission_path
):

    df_kaggle_test = pl.read_csv(kaggle_test_path)
    kaggle_sample_submission = pl.read_csv(kaggle_sample_submission_path)

    kaggle_test_preds = pipeline.predict(df_kaggle_test)
    kaggle_sample_submission = kaggle_sample_submission.with_columns(
        pl.Series("target", kaggle_test_preds)
    )

    submissions_dir = Path("data/submissions")
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    filename = f"submission_{timestamp}.csv"
    submission_path = submissions_dir / filename

    kaggle_sample_submission.write_csv(submission_path)

    kaggle_score = submit_to_kaggle("nlp-getting-started", submission_path)
    logging.info(f"Public Kaggle score: {kaggle_score}")
    mlflow.log_metric("public_kaggle_score", kaggle_score)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run_experiment(cfg):

    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(nested=True):
        mlflow.set_tags(OmegaConf.to_container(cfg.mlflow.tags))
        mlflow.log_params(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))

        df_train, df_val, df_test = load_or_create_data(cfg)

        if cfg.experiment.tuning:
            # Hyperparameter tuning is enabled.
            # Pass the new hyperparameter parameters structure from config to the tuner.
            best_model, best_params, best_metric = tune_hyperparameters(
                df_train,
                df_val,
                cfg.model_params,
                OmegaConf.to_container(cfg.hyperparameter_tuning.parameters),
                n_trials=cfg.hyperparameter_tuning.n_trials,
            )
            for key, value in best_params.items():
                mlflow.log_param(f"best_model_params.{key}", value)
            mlflow.log_metric("best_val_roc_auc", best_metric)
            pipeline_model = best_model
        else:
            pipeline_model = train(df_train, df_val, model_params=cfg.model_params)

        eval_df_test(pipeline_model, df_test)

        if cfg.experiment.submit_to_kaggle:
            df_full_train = pl.concat([df_train, df_val, df_test])
            full_pipeline = train(
                df_full_train, model_params=cfg.model_params, full_train=True
            )
            generate_and_submit_to_kaggle(
                full_pipeline,
                cfg.raw_data.test_path,
                cfg.raw_data.sample_submission_path,
            )


if __name__ == "__main__":
    run_experiment()
