"""
This module provides utility functions for data manipulation and Kaggle competition submissions.

Functions:
- polars_train_val_test_split(df: pl.DataFrame, train_frac: float = 0.8, val_frac: float = 0.1, test_frac: float = 0.1, shuffle: bool = True, seed: int = 42) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:

- submit_to_kaggle(competition_name: str, submission_file: str, message: str = "Submission"):
"""

import logging
import os
from typing import Tuple

import kaggle
import polars as pl


def polars_train_val_test_split(
    df: pl.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Splits a Polars DataFrame into train, validation, and test sets.

    Parameters:
    - df (pl.DataFrame): The input DataFrame.
    - train_frac (float): Fraction of data for training (default 0.7).
    - val_frac (float): Fraction of data for validation (default 0.15).
    - test_frac (float): Fraction of data for testing (default 0.15).
    - shuffle (bool): Whether to shuffle the data before splitting (default True).
    - seed (int): Random seed for shuffling (default 42).

    Returns:
    - (pl.DataFrame, pl.DataFrame, pl.DataFrame): Train, Validation, and Test DataFrames.
    """
    assert train_frac + val_frac + test_frac == 1.0, "Fractions must sum to 1"

    if shuffle:
        df = df.sample(fraction=1.0, shuffle=True, seed=seed)

    train_end = int(train_frac * len(df))
    val_end = train_end + int(val_frac * len(df))

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


def submit_to_kaggle(
    competition_name: str, submission_file: str, message: str = "Submission"
):
    """
    Submits a file to a Kaggle competition.

    Args:
        competition_name (str): The competition name (e.g., "titanic").
        submission_file (str): The path to the submission file.
        message (str): Submission message.
    """
    try:
        kaggle.api.competition_submit(
            file_name=submission_file, competition=competition_name, message=message
        )
        print(
            f"Successfully submitted {submission_file} to {competition_name} with message: '{message}'"
        )
    except Exception as e:
        print(f"Submission failed: {e}")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
