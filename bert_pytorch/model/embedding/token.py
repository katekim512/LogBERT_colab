import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512, sbert_weights=None):
        super().__init__(vocab_size, embed_size, padding_idx=0)

        if sbert_weights is not None:
            print("Injecting SBERT Semantic Weights into TokenEmbedding...")
            self.weight.data.copy_(torch.from_numpy(sbert_weights).float())
