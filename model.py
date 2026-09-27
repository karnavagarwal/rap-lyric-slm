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
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        head_dim = C//self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1,2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        #divided by 8 just to scale it down
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class RapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(30000, 768)
        self.pos_emb = nn.Embedding(1024, 768)
        self.blocks = nn.Sequential(*[Block() for _ in range(12)])
        self.ln_f = nn.LayerNorm(768)
        self.lm_head = nn.Linear(768, 30000, bias = False)

    def forward(self, x):
        B, T = x.size()
        #x is the batch of token IDs.
        #B will be the number of sequences of token IDs.
        #T will be the sequence length
        positions = torch.arange(0, T, dtype = torch.long, device = x.device)
        tok_vec = self.token_emb(x)
        pos_vec = self.pos_emb(positions)
        x = tok_vec + pos_vec
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(768, 4 * 768)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * 768, 768)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(768)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(768)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


if __name__ == "__main__":
    dummy_input = torch.randint(0, 30000, (4, 128))
    model = RapModel()
    output = model(dummy_input)
    print("Input shape: ", dummy_input.shape)
    print("Output shape: ", output.shape)