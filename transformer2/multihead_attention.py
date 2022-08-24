import torch
from torch import nn, Tensor
import torch.nn.functional as f
from scaled_dot_product_attention import scaled_dot_product_attention
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.nn.parameter import Parameter
from typing import Optional, Tuple


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dropout_p: float = 0.0,
        kdim: int = 0,
        vdim: int = 0,
        bias: bool = True,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()

        self.dim_model = dim_model
        self.kdim = kdim or dim_model
        self.vdim = vdim or dim_model

        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.head_dim = dim_model // n_heads
        assert (
            dim_model % n_heads == 0
        ), f"dim_model ({dim_model}) not divisible by head_dim ({self.head_dim})"

        self.q_proj_w = Parameter(torch.empty((dim_model, dim_model), device=device))
        self.k_proj_w = Parameter(torch.empty((dim_model, self.kdim), device=device))
        self.v_proj_w = Parameter(torch.empty((dim_model, self.vdim), device=device))
        self.out_proj_w = Parameter(torch.empty((dim_model, dim_model), device=device))

        if bias is not None:
            self.bias_q = Parameter(torch.empty((1, dim_model), device=device))
            self.bias_k = Parameter(torch.empty((1, dim_model), device=device))
            self.bias_v = Parameter(torch.empty((1, dim_model), device=device))
            self.bias_out_proj = Parameter(torch.empty((1, dim_model), device=device))
        else:
            self.bias_q = self.bias_k = self.bias_v = self.bias_out_proj = None

        self._init_parameters()

    def _init_parameters(self):
        xavier_uniform_(self.q_proj_w)
        xavier_uniform_(self.k_proj_w)
        xavier_uniform_(self.v_proj_w)
        xavier_uniform_(self.out_proj_w)

        if (
            self.bias_q is not None
            and self.bias_k is not None
            and self.bias_v is not None
            and self.bias_out_proj is not None
        ):
            xavier_normal_(self.bias_q)
            xavier_normal_(self.bias_k)
            xavier_normal_(self.bias_v)
            constant_(self.bias_out_proj, 0)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q: tensor, shape[B, Nt, Dq]
            k: tensor, shape[B, Ns, Dk]
            v: tensor, shape[B, Ns, Dv]
            attn_mask: tensor, shape[Nt, Ns]

        Returns:
            attn_output: tensor, shape[B, Nt, Dq]
            attn_weights: tensor, shape[B, n_heads Nt, Ns]
        """

        # (B, N, D) -> (N, B, D)
        q, k, v = [x.transpose(1, 0) for x in (q, k, v)]
        p = self.dropout_p if self.training else 0.0

        Nt, B, Dq = q.shape
        Ns, _, _ = k.shape
        assert Dq == self.dim_model, f"dim_q should be {self.dim_model}, but got {Dq}"

        q = f.linear(q, self.q_proj_w, self.bias_q)  # (Nt, B, Dq) -> (Nt, B, Dq)
        k = f.linear(k, self.k_proj_w, self.bias_k)  # (Ns, B, Dk) -> (Ns, B, Dq)
        v = f.linear(v, self.v_proj_w, self.bias_v)  # (Ns, B, Dv) -> (Ns, B, Dq)

        if attn_mask is not None:
            assert (
                attn_mask.is_floating_point()
            ), f"attn_mask must be of type float, reveived {attn_mask.dtype}"

        # (N, B, D) -> (B * n_heads, N, head_dim)
        q = q.contiguous().view(Nt, B * self.n_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(Ns, B * self.n_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(Ns, B * self.n_heads, self.head_dim).transpose(0, 1)

        # (B * n_heads, Nt, head_dim), (B * n_heads, Nt, Ns)
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, attn_mask, p)

        # (B * n_heads, Nt, head_dim) -> (B * Nt, D)
        attn_output = attn_output.transpose(0, 1).contiguous().view(Nt * B, Dq)

        # (B * Nt, D) x (D, D) -> (B * Nt, D)
        attn_output = f.linear(attn_output, self.out_proj_w, self.bias_out_proj)

        # (B * Nt, D) -> (B, Nt, D)
        attn_output = attn_output.view(B, Nt, Dq)

        # (B * n_heads, Nt, Ns) -> (B, n_heads, Nt, Ns)
        attn_weights = attn_weights.view(B, self.n_heads, Nt, Ns)

        return attn_output, attn_weights


def main():
    B = 16
    n_heads = 8
    Nt, Ns = 40, 20
    Dq, Dk, Dv = 512, 256, 128

    q = torch.rand(B, Nt, Dq)
    k = torch.rand(B, Ns, Dk)
    v = torch.rand(B, Ns, Dv)

    print(f"q: {q.shape}\nk: {k.shape}\nv: {v.shape}")
    print()

    multihead_attention = MultiheadAttention(Dq, n_heads=n_heads, kdim=Dk, vdim=Dv)

    attn_output, attn_weights = multihead_attention(q, k, v)
    print(f"attn_output: {attn_output.shape}")
    print(f"attn_weights: {attn_weights.shape}")


if __name__ == "__main__":
    main()
