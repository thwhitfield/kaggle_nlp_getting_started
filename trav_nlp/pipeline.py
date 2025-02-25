import datetime
import logging
from pathlib import Path

import hydra
import lightgbm as lgb
import polars as pl
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from trav_nlp.misc import polars_train_val_test_split, submit_to_kaggle


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


def train(df_train, df_val=None):
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
        lgb.LGBMClassifier(random_state=42),
    )

    pipeline.fit(df_train, df_train["target"])

    train_preds = pipeline.predict_proba(df_train)[:, 1]
    train_roc_auc = roc_auc_score(df_train["target"], train_preds)
    logging.info(f"Train ROC: {train_roc_auc}")

    if df_val is not None:
        val_preds = pipeline.predict_proba(df_val)[:, 1]
        val_roc_auc = roc_auc_score(df_val["target"], val_preds)
        logging.info(f"Val ROC: {val_roc_auc}")

    return pipeline


def eval_df_test(pipeline, df_test):

    test_preds = pipeline.predict_proba(df_test)[:, 1]
    test_roc_auc = roc_auc_score(df_test["target"], test_preds)
    logging.info(f"Test ROC: {test_roc_auc}")


def generate_and_submit_to_kaggle(
    pipeline, kaggle_test_path, kaggle_sample_submission_path
):

    df_kaggle_test = pl.read_csv(kaggle_test_path)
    kaggle_sample_submission = pl.read_csv(kaggle_sample_submission_path)

    kaggle_test_preds = pipeline.predict(df_kaggle_test)
    kaggle_sample_submission = kaggle_sample_submission.with_columns(
        pl.Series("target", kaggle_test_preds)
    )

    submissions_dir = Path("../data/submissions")
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    filename = f"submission_{timestamp}.csv"
    submission_path = submissions_dir / filename

    kaggle_sample_submission.write_csv(submission_path)

    submit_to_kaggle("nlp-getting-started", submission_path)


@hydra.main(config_path="../conf", config_name="config")
def run_experiment(cfg):

    df_train, df_val, df_test = load_or_create_data(cfg)

    pipeline = train(df_train, df_val)

    eval_df_test(pipeline, df_test)

    if cfg.experiment.submit_to_kaggle:
        df_full_train = pl.concat([df_train, df_val, df_test])
        full_pipeline = train(df_full_train)
        generate_and_submit_to_kaggle(
            full_pipeline, cfg.raw_data.test_path, cfg.raw_data.sample_submission_path
        )


if __name__ == "__main__":
    run_experiment()
