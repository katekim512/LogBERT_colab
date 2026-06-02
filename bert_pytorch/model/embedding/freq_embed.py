import torch.nn as nn


class FreqEmbedding(nn.Module):
    """
    Frequency(=count) scalar -> embedding vector
    Input shape: (batch, seq_len, 1)
    Output shape: (batch, seq_len, embed_size)
    """

    def __init__(self, embed_size=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

    def forward(self, freq_value):
        return self.mlp(freq_value)
