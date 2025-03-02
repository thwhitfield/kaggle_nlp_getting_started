"""
This module provides utility functions for data manipulation and Kaggle competition submissions.

Functions:
- polars_train_val_test_split(df: pl.DataFrame, train_frac: float = 0.8, val_frac: float = 0.1, test_frac: float = 0.1, shuffle: bool = True, seed: int = 42) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:

- submit_to_kaggle(competition_name: str, submission_file: str, message: str = "Submission"):
"""

import logging
import os
import subprocess
import time
from typing import Tuple

import kaggle
import polars as pl


def polars_train_val_test_split(
    df: pl.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    shuffle: bool = True,
    seed: int = None,
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


def polars_train_test_split(
    df: pl.DataFrame,
    train_frac: float = 0.8,
    test_frac: float = 0.2,
    shuffle: bool = True,
    seed: int = None,
) -> "Tuple[pl.DataFrame, pl.DataFrame]":
    """
    Splits a Polars DataFrame into train and test sets.

    Parameters:
    - df (pl.DataFrame): The input DataFrame.
    - train_frac (float): Fraction of data for training.
    - test_frac (float): Fraction of data for testing.
    - shuffle (bool): Whether to shuffle the data before splitting.
    - seed (int): Random seed for shuffling.

    Returns:
    - (pl.DataFrame, pl.DataFrame): Train and Test DataFrames.
    """
    assert train_frac + test_frac == 1.0, "Fractions must sum to 1"

    if shuffle:
        df = df.sample(fraction=1.0, shuffle=True, seed=seed)

    n = len(df)

    train_end = int(train_frac * n)
    train_df = df[:train_end]
    test_df = df[train_end:]
    return train_df, test_df


def submit_to_kaggle(
    competition_name: str,
    submission_file: str,
    message: str = "Submission",
    wait_for_score: bool = True,
    timeout_mins: int = 10,
):
    """
    Submits a file to a Kaggle competition and optionally waits for the score.

    Args:
        competition_name (str): The competition name (e.g., "titanic").
        submission_file (str): The path to the submission file.
        message (str): Submission message.
        wait_for_score (bool): Whether to wait for and return the score.
        timeout_mins (int): Maximum time to wait for score in minutes.

    Returns:
        float or None: The submission score if wait_for_score is True and score is available, None otherwise.
    """

    try:
        # Submit the file
        kaggle.api.competition_submit(
            file_name=submission_file, competition=competition_name, message=message
        )
        print(
            f"Successfully submitted {submission_file} to {competition_name} with message: '{message}'"
        )

        if not wait_for_score:
            return None

        print(f"Waiting for score (timeout: {timeout_mins} minutes)...")

        # Get the submission ID
        submissions = kaggle.api.competition_submissions(competition_name)
        submission_id = None

        for submission in submissions:
            if submission.description == message:
                submission_id = submission.ref
                break

        if not submission_id:
            print("Could not find submission ID for the just-submitted file.")
            return None

        # Wait for the score with timeout
        start_time = time.time()
        timeout_secs = timeout_mins * 60

        while time.time() - start_time < timeout_secs:
            submissions = kaggle.api.competition_submissions(competition_name)

            for submission in submissions:
                if (
                    submission.ref == submission_id
                    and submission.status.lower() == "complete"
                ):
                    score = float(submission.publicScore)
                    print(f"Submission scored: {score}")
                    return score

            # Wait before checking again
            time.sleep(30)
            print("Still waiting for score...")

        print(f"Timed out after {timeout_mins} minutes waiting for score.")
        return None

    except Exception as e:
        print(f"Submission failed: {e}")
        return None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def flatten_dict(d, parent_key="", sep="."):
    """Flatten a nested dictionary.

    Args:
        d (_type_): _description_
        parent_key (str, optional): _description_. Defaults to ''.
        sep (str, optional): _description_. Defaults to '.'.

    Returns:
        _type_: _description_
    """

    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def verify_git_commit(*target_folders):
    """
    Verifies that each specified folder within a git repository has no uncommitted changes or untracked files.

    Parameters:
        *target_folders (str): One or more paths to folders within your repository that you want to check.

    Returns:
        str: The commit hash of HEAD if the working tree is clean across all folders.

    Raises:
        RuntimeError: If any folder is not within a git repository, if folders are in different repositories,
                      or if there are local changes or untracked files in any of the folders.
    """
    if not target_folders:
        raise ValueError("At least one target folder must be specified.")

    repo_root = None

    # Check each target folder.
    for folder in target_folders:
        # Convert folder path to an absolute path.
        abs_folder = os.path.abspath(folder)

        # Determine the repository root for the current folder.
        try:
            current_repo = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], cwd=abs_folder, text=True
            ).strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"The specified folder '{abs_folder}' is not inside a git repository."
            ) from e

        # Ensure all folders belong to the same repository.
        if repo_root is None:
            repo_root = current_repo
        elif repo_root != current_repo:
            raise RuntimeError(
                f"All target folders must be in the same git repository. "
                f"'{abs_folder}' is in a different repository than '{repo_root}'."
            )

        # Get the relative path of the folder with respect to the repo root.
        rel_folder = os.path.relpath(abs_folder, repo_root)

        # Check for uncommitted changes in files tracked by git.
        diff_cmd = ["git", "diff", "--exit-code", "HEAD", "--", rel_folder]
        try:
            subprocess.check_call(
                diff_cmd,
                cwd=repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            raise RuntimeError(f"There are uncommitted changes in '{abs_folder}'.")

        # Check for untracked files within the target folder.
        ls_files_cmd = [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            rel_folder,
        ]
        untracked = subprocess.check_output(
            ls_files_cmd, cwd=repo_root, text=True
        ).strip()
        if untracked:
            raise RuntimeError(f"There are untracked files in '{abs_folder}'.")

    # If all checks pass, return the current commit hash.
    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()

    return commit_hash


def get_git_commit_hash():
    # If all checks pass, return the current commit hash.
    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    return commit_hash
