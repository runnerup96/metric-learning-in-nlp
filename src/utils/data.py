import json
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split


def load_dataframe(path: str) -> pd.DataFrame:
    """Load json dataset provided by
    https://github.com/clinc/oos-eval.git and convert it to DataFrame

    Args:
        path (str): path to json, e.g. oos-eval/data/data_full.json

    Returns:
        pd.DataFrame: total dataframe
    """
    with open(path, "r") as f:
        data_full = json.load(f)

    dataset_list = [
        pd.DataFrame(data_full[key], columns=["intents", "targets"])
        for key in data_full.keys()
    ]
    total_df = pd.concat(dataset_list, axis=0, sort=False).reset_index(drop=True)
    return total_df


def split_by_classes(
    total_df: pd.DataFrame, seed: int = 42, train_classes_ratio: float = 0.6
) -> Dict[str, pd.DataFrame]:
    """Perform train, val, test (seen/unseen) split.
    Assume train_classes_ratio = 0.6, then ratios will be the following:
        * train - 0.6
        * val - 0.2
        * test - 0.2

    Args:
        total_df (pd.DataFrame): dataframe to split.
        seed (int, optional): Defaults to 42.
        train_classes_ratio (float, optional): _description_. Defaults to 0.6.

    Returns:
        Dict[str, pd.DataFrame]: Dict with keys:
            `(train_df, val_df, test_df, seen_test_df, unseen_test_df, test_classes)`.
    """
    classes_list = list(set(total_df["targets"].tolist()))

    # Perform split by class to (train, val, test)
    train_classes, __eval_classes = train_test_split(
        classes_list, train_size=train_classes_ratio, random_state=seed
    )
    val_classes, test_classes = train_test_split(
        __eval_classes, train_size=0.5, random_state=seed
    )

    # Select sub-dataframes
    train_df = total_df[total_df["targets"].isin(train_classes)]
    val_df = total_df[total_df["targets"].isin(val_classes)]
    test_df = total_df[total_df["targets"].isin(test_classes)]

    # Also, its good idea to split test to seen/unseen groups
    X_test_seen, X_test_unseen, y_test_seen, y_test_unseen = train_test_split(
        test_df["intents"].to_list(),
        test_df["targets"].to_list(),
        test_size=0.5,
        random_state=seed,
    )
    seen_test_df = pd.DataFrame({"intents": X_test_seen, "targets": y_test_seen})
    unseen_test_df = pd.DataFrame({"intents": X_test_unseen, "targets": y_test_unseen})

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "seen_test_df": seen_test_df,
        "unseen_test_df": unseen_test_df,
        "test_classes": test_classes,
    }


def validate_splits(splits: Dict[str, pd.DataFrame]) -> None:
    # classes from train & val does not intersect
    assert not set(splits["train_df"]["targets"].tolist()).intersection(
        splits["val_df"]["targets"].tolist()
    )

    # classes from train & test does not intersect
    assert not set(splits["train_df"]["targets"].tolist()).intersection(
        splits["test_df"]["targets"].tolist()
    )

    # classes from val & test does not intersect
    assert not set(splits["val_df"]["targets"].tolist()).intersection(
        splits["test_df"]["targets"].tolist()
    )

    # classes from seen test & unseen test intersect fully
    assert len(
        set(splits["seen_test_df"]["targets"].tolist()).intersection(
            splits["unseen_test_df"]["targets"].tolist()
        )
    ) == len(splits["test_classes"])
