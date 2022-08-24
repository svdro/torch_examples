import copy
import torch
import torch.nn.functional as f

from multihead_attention import MultiheadAttention
from torch import nn, Tensor
from typing import Callable, Optional, Tuple, List


def _get_clones(module: nn.Module, n_clones) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_clones)])


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feed_forward: int,
        dropout_p: float = 0.0,
        activation: Callable[[Tensor], Tensor] = f.relu,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()

        self.self_attn = MultiheadAttention(
            dim_model, n_heads, dropout_p, device=device
        )

        self.linear1 = nn.Linear(dim_model, dim_feed_forward, device=device)
        self.linear2 = nn.Linear(dim_feed_forward, dim_model, device=device)

        eps = 1e-5
        self.norm1 = nn.LayerNorm(dim_model, eps=eps, device=device)
        self.norm2 = nn.LayerNorm(dim_model, eps=eps, device=device)

        self.dropout_attn = nn.Dropout(dropout_p)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

        self.activation = activation

    def _self_attention_block(
        self, x: Tensor, attn_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        x, attn_weights = self.self_attn(x, x, x, attn_mask)
        return self.dropout_attn(x), attn_weights

    def _feed_forward_block(self, x: Tensor) -> Tensor:
        x = self.dropout1(self.linear1(self.activation(x)))
        return self.dropout2(self.linear2(x))

    def forward(
        self, src: Tensor, src_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            src: tensor, shape[B, N, Dq]
            src_mask: tensor, shape[N, N]

        Returns:
            output: tensor, shape[B, N, Dq]
            attn_weights: tensor, shape[B, n_heads, N, N]
        """

        # self attention black
        x, attn_weights = self._self_attention_block(src, src_mask)
        x = self.norm1(src + x)

        # feed_forward_block
        x = x + self._feed_forward_block(x)
        x = self.norm2(x)

        return x, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, n_layers: int):
        super().__init__()
        self.layers = _get_clones(encoder_layer, n_layers)
        self.n_layers = n_layers

    def forward(
        self, src: Tensor, src_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Args:
            src: tensor, shape[B, N, Dq]
            src_mask: list of tensors, shape[n_layers [B, n_heads, N, N] ]
        """
        x = src

        attn_weights = []
        for layer in self.layers:
            x, attn_w = layer(x, src_mask=src_mask)
            attn_weights.append(attn_w)

        return x, attn_weights


def main():
    B, N = 2, 10
    Dff = 2048
    D = 512
    n_heads = 4

    src = torch.rand(B, N, D)
    encoderLayer = TransformerEncoderLayer(D, n_heads, Dff)
    encoder = TransformerEncoder(encoderLayer, 2)

    output, attn_weights = encoder(src)
    print(output.shape)
    print([attn_w.shape for attn_w in attn_weights])


if __name__ == "__main__":
    main()
