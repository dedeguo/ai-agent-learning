"""可选 PyTorch Pre-LN 示例：运行前需安装 torch。"""
import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=4, heads=2, d_ff=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=0., batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x):  # (batch, tokens, d_model)
        z = self.ln1(x)
        a, weights = self.attn(z, z, z, need_weights=True, average_attn_weights=False)
        h = x + a  # (batch, tokens, d_model)
        y = h + self.ffn(self.ln2(h))
        return y, weights  # weights: (batch, heads, tokens, tokens)

if __name__ == "__main__":
    torch.manual_seed(3)
    model = TransformerBlock().eval()
    with torch.no_grad():
        y, weights = model(torch.randn(1, 3, 4))
    print("output:", y.shape, "weights:", weights.shape)
