from torch import nn, Tensor
from torch.nn import Transformer
from utils import TokenEmbedding, PositionalEncoding


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        n_encoder_layers: int,
        n_decoder_layers: int,
        d_model: int,
        n_heads: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_feedforward: int = 512,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.transformer = Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_feedforward,
            dropout=dropout_p,
            batch_first=True,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout_p)

        self._init_parameters()

    def _init_parameters(self):
        """initialize all parameters except biases with xavier_uniform_"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        """
        Args:
            src: tensor(torch.int64), shape[B, Ns]
            tgt: tensor(torch.int64), shape[B, Nt]
        Returns:
            outs: tensor, shape[B, Nt, tgt_vocab_size]

        """
        # (B, Ns, D), (B, Nt, D)
        src_emb = self.pe(self.src_tok_emb(src))
        tgt_emb = self.pe(self.tgt_tok_emb(tgt))

        # (B, Nt, D)
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )

        # (B, Nt, tgt_vocab_size)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: tensor(torch.int64), shape[B, Ns]
            src_mask: tensor, shape[Ns, Ns]

        Returns:
            memory: tensor, shape[B, Ns, D]
        """
        return self.transformer.encoder(self.pe(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        """
        Args:
            tgt: tensor(torch.int64), shape[B, Nt]
            memory: tensor, shape[B, Ns, D]
            tgt_mask: tensor, shape[Nt, Nt]

        Returns:
            out: tensor, shape[B, Nt, D]
        """
        return self.transformer.decoder(
            self.pe(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


def main():
    import torch
    from utils import create_masks

    src_vocab_size, tgt_vocab_size = 10000, 10000
    n_layers, n_heads = 1, 2
    D, Dff = 8, 32
    B, Ns, Nt = 2, 10, 12

    src = torch.rand(B, Ns)
    tgt = torch.rand(B, Nt)

    transformer = Seq2SeqTransformer(
        n_layers, n_layers, D, n_heads, src_vocab_size, tgt_vocab_size, Dff, 0
    )

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(src, tgt, 1)
    print(f"src: {src.shape}, {src.dtype}\ntgt: {tgt.shape}, {tgt.dtype}")
    print(f"src_mask: {src_mask.shape}, {src_mask.dtype}")
    print(f"tgt_mask: {tgt_mask.shape}, {tgt_mask.dtype}")
    print(f"src_padding_mask: {src_padding_mask.shape}, {src_padding_mask.dtype}")
    print(f"tgt_padding_mask: {tgt_padding_mask.shape}, {tgt_padding_mask.dtype}\n")

    memory = transformer.encode(src, src_mask)
    out = transformer.decode(tgt, memory, tgt_mask)

    logits = transformer(
        src,
        tgt,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
        src_padding_mask,
    )

    print(f"memory: {memory.shape}, {memory.dtype}")
    print(f"out: {out.shape}, {out.dtype}")
    print(f"logits: {logits.shape}, {logits.dtype}")


if __name__ == "__main__":
    main()
