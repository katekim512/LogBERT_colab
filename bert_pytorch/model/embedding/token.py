import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512, sbert_weights=None):
        super().__init__(vocab_size, embed_size, padding_idx=0)

        # SBERT 가중치가 전달되었다면 초기값으로 설정
        if sbert_weights is not None:
            print("Injecting SBERT Semantic Weights into TokenEmbedding...")
            # numpy 배열을 torch 텐서로 변환하여 복사
            self.weight.data.copy_(torch.from_numpy(sbert_weights).float())
            
            # (옵션) 만약 의미를 고정하고 싶다면 아래 주석 해제
            # self.weight.requires_grad = False
