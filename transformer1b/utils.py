import math
import torch
from torch import nn, Tensor
from typing import Tuple


def generate_square_subsequent_mask(
    N: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tensor:
    """
    Returns:
        mask: tensor, shape[N, N]
    """
    mask = (torch.triu(torch.ones((N, N), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_masks(
    src: Tensor,
    tgt: Tensor,
    pad_idx: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Args:
        src: tensor(torch.int64), shape[B, Ns]
        tgt: tensor(torch.int64), shape[B, Nt]

    Returns:
        src_mask: tensor(torch.float32), shape[Ns, Ns]
        tgt_mask: tensor(torch.float32), shape[Nt, Nt]
        src_padding_mask: tensor(torch.bool), shape[B, Ns]
        tgt_padding_mask: tensor(torch.bool), shape[B, Ns]
    """
    Ns, Nt = src.shape[1], tgt.shape[1]

    src_mask = torch.zeros((Ns, Ns), device=device).type(torch.float)
    tgt_mask = generate_square_subsequent_mask(Nt, device)

    src_padding_mask = src == pad_idx
    tgt_padding_mask = tgt == pad_idx
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_p: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]

        Returns:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x = x + self.pe[: x.size(0)]
        # because nvim diagnostics does not play well with "register_buffer"
        pe = self._buffers["pe"]
        assert isinstance(pe, Tensor)

        x = x + pe[: x.size(0)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: tensor(torch.int64), shape[B, N]

        Returns:
            embedding: tensor(torch.float32), shape[B, N, D]
        """
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


if __name__ == "__main__":
    vocab_size = int(1e4)
    B, Ns, Nt, D = 4, 10, 12, 2
    embedding = TokenEmbedding(vocab_size, D)
    pe = PositionalEncoding(D)

    src = torch.randint(0, vocab_size, (B, Ns))
    tgt = torch.randint(0, vocab_size, (B, Nt))
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(src, tgt, 1)

    print(f"src: {src.shape}, {src.dtype}\ntgt:{tgt.shape}, {tgt.dtype}")
    print(f"src_mask: {src_mask.shape}, {src_mask.dtype}")
    print(f"tgt_mask:{tgt_mask.shape}, {tgt_mask.dtype}")
    print(f"src_padding_mask: {src_padding_mask.shape}, {src_padding_mask.dtype}")
    print(f"tgt_padding_mask:{tgt_padding_mask.shape}, {tgt_padding_mask.dtype}")

    emb = embedding(src)
    pe_ = pe(emb)
    print(f"embedding: ", emb.shape)
    print(f"pe: ", pe_.shape)
