import pandas as pd
import torch

from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class PairDataset(Dataset):
    def __init__(self, train_file_path, vocab, max_len=50):
        self.train_file = pd.read_csv(train_file_path)
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.train_file)

    def __getitem__(self, idx):
        question1 = self.train_file.iloc[idx]["question1"]
        question2 = self.train_file.iloc[idx]["question2"]
        target = self.train_file.iloc[idx]["is_duplicate"]

        q1_tokenized_tensor = self._tokenize(question1)
        q2_tokenized_tensor = self._tokenize(question2)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return q1_tokenized_tensor, q2_tokenized_tensor, target_tensor

    def _tokenize(self, text):
        token_ids = []

        for word in str(text).split():
            token_id = self.vocab.get(word, self.vocab[UNK_TOKEN])
            token_ids.append(token_id)

        token_ids = token_ids[:self.max_len]

        padding_size = self.max_len - len(token_ids)
        token_ids += [self.vocab[PAD_TOKEN]] * padding_size

        return torch.tensor(token_ids, dtype=torch.long)
