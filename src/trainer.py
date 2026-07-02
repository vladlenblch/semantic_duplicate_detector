from collections import Counter
from pathlib import Path
import pandas as pd

from torch.utils.data import DataLoader

from src.data.dataset import PairDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEXT_COLUMNS = ["question1", "question2"]

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_VOCAB_SIZE = 100_000


def build_vocab(train_file_path, max_vocab_size=MAX_VOCAB_SIZE):
    train_df = pd.read_csv(train_file_path)
    word_counts = Counter()

    for col in TEXT_COLUMNS:
        words = train_df[col].str.split()
        word_counts.update(word for row in words for word in row)

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    max_words = max_vocab_size - len(vocab)
    most_common_words = word_counts.most_common(max_words)

    for word, _ in most_common_words:
        vocab[word] = len(vocab)

    return vocab


def main():
    vocab = build_vocab(PROCESSED_DATA_PATH)

    train_dataset = PairDataset(
        train_file_path=PROCESSED_DATA_PATH,
        vocab=vocab,
        max_len=50
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )


if __name__ == "__main__":
    main()
