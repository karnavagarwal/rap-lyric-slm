import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head = 12
        self.n_embd = 768
        #Q, K, V
        self.c_attn = nn.Linear(768, 3*768)
        self.c_proj = nn.Linear(768, 768)
        self.register_buffer("bias", torch.tril(torch.ones(1024,1024)).view(1,1,1024,1024))

    def forward(self, x):
        pass

class RapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(30000, 768)
        self.pos_emb = nn.Embedding(1024, 768)

    def forward(self, x):
        B, T = x.size()
        #x is the batch of token IDs.
        #B will be the number of sequences of token IDs.
        #T will be the sequence length
        positions = torch.arange(0, T, dtype = torch.long)
        tok_vec = self.token_emb(x)
        pos_vec = self.pos_emb(positions)
        return tok_vec + pos_vec

if __name__ == "__main__":
    dummy_input = torch.randint(0, 30000, (4, 128))
    model = RapModel()
    output = model(dummy_input)
    print("Input shape: ", dummy_input.shape)
    print("Output shape: ", output.shape)