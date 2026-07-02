from torch import nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx=0):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.padding_idx = padding_idx

        
    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)

        mask = input_ids != self.padding_idx
        masked_embeddings = embeddings * mask.unsqueeze(-1)

        summed_embeddings = masked_embeddings.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)

        pooled = summed_embeddings / lengths
        output = self.projection(pooled)

        return F.normalize(output, p=2, dim=1)
