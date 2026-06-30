import html
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train_raw.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

COLUMNS_TO_DROP = ["id", "qid1", "qid2"]

FEATURES_TO_PREPROCESS = ["question1", "question2"]

HTML_TAG_PATTERN = re.compile(r"<.*?>")
SPECIAL_CHARS_PATTERN = re.compile(r"[^\w\s]")
UNDERSCORE_PATTERN = re.compile(r"_+")
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")

TARGET_COLUMN = "is_duplicate"
RANDOM_STATE=42


def load_raw_data():
    df = pd.read_csv(RAW_DATA_PATH)

    return df


def remove_invalid_rows(df):
    cleaned_df = df.dropna(subset=FEATURES_TO_PREPROCESS)

    return cleaned_df


def remove_useless_features(df):
    cleaned_df = df.drop(columns=COLUMNS_TO_DROP)

    return cleaned_df


def to_lowercase(df):
    for col in FEATURES_TO_PREPROCESS:
        df[col] = df[col].str.lower()

    return df


def clean_text(text):
    if not isinstance(text, str):
        return text

    text = html.unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = SPECIAL_CHARS_PATTERN.sub(" ", text)
    text = UNDERSCORE_PATTERN.sub(" ", text)
    text = MULTIPLE_SPACES_PATTERN.sub(" ", text)
    text = text.strip()

    return text


def clean_data(df):
    for col in FEATURES_TO_PREPROCESS:
        df[col] = df[col].map(clean_text)

    return df


def preprocess_data(df):
    df = df.copy()

    df = remove_invalid_rows(df)
    df = remove_useless_features(df)
    df = to_lowercase(df)
    df = clean_data(df)

    return df


def stratified_split(df):
    train_df, tmp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df[TARGET_COLUMN]
    )

    val_df, test_df = train_test_split(
        tmp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=tmp_df[TARGET_COLUMN]
    )

    return train_df, val_df, test_df


def save_data(train_df, val_df, test_df):
    train_df.to_csv(f"{PROCESSED_DATA_PATH}/train.csv", index=False)
    val_df.to_csv(f"{PROCESSED_DATA_PATH}/val.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DATA_PATH}/test.csv", index=False)


def main():
    raw_df = load_raw_data()
    processed_df = preprocess_data(raw_df)
    train_df, val_df, test_df = stratified_split(processed_df)
    save_data(train_df, val_df, test_df)
    print(f"Data successfully preprocessed")


if __name__ == "__main__":
    main()
