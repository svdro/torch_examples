import math
import torch
import torch.nn.functional as f
from torch import nn, Tensor
from typing import Optional, Tuple


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """
    B is batch size, Nt is target length, Ns is source length, D is feature dimenstions.

    Args:
        q: tensor, shape [B, Nt, D]
        k: tensor, shape [B, Ns, D]
        v: tensor, shape [B, Ns, D]
        attn_mask: tensor, shape [Nt, Ns]

    Returns:
        out: tensor, shape [B, Nt, D]
        attn: tensor, shape [B, Nt, Ns]
    """
    # _, _, D = q.shape
    D = q.size(-1)
    q = q / math.sqrt(D)

    # (B, Nt, D) x (B, D, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask

    attn = f.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = f.dropout(attn, p=dropout_p)

    # (B, Nt, Ns) x (B, Ns, D) -> (B, Nt, D)
    out = torch.bmm(attn, v)
    return out, attn


def main():
    B, Nt, Ns, D = 1, 20, 25, 512
    q = torch.rand(B, Nt, D)
    k, v = [torch.rand(B, Ns, D) for _ in range(2)]

    out, attn = scaled_dot_product_attention(q, k, v)
    print(f"out: {out.shape}\nattn: {attn.shape}")


if __name__ == "__main__":
    main()
