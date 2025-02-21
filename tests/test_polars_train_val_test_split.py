import polars as pl
import pytest
from trav_nlp.misc import polars_train_val_test_split  # fixed import


def test_polars_train_val_test_split():
    # Create a small dataframe with 10 rows
    df = pl.DataFrame({"feature": list(range(10)), "label": [i % 2 for i in range(10)]})

    # Use train=0.6, val=0.2, test=0.2 (sums to 1.0)
    train_df, val_df, test_df = polars_train_val_test_split(
        df, train_frac=0.6, val_frac=0.2, test_frac=0.2, shuffle=False
    )

    total_rows = len(df)
    expected_train = int(0.6 * total_rows)
    expected_val = int(0.2 * total_rows)
    expected_test = (
        total_rows - expected_train - expected_val
    )  # account for integer rounding

    assert len(train_df) == expected_train, "Train split size mismatch"
    assert len(val_df) == expected_val, "Validation split size mismatch"
    assert len(test_df) == expected_test, "Test split size mismatch"
    assert (
        len(train_df) + len(val_df) + len(test_df) == total_rows
    ), "Total rows mismatch"


if __name__ == "__main__":
    pytest.main([__file__])
