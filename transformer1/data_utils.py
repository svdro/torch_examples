import torch
from torch import Tensor
from torch.utils.data import dataset
from torchtext.vocab import Vocab
from typing import Any


def data_process(
    raw_text_iter: dataset.IterableDataset, vocab: Vocab, tokenizer: Any
) -> Tensor:
    """Converts raw text into a flat tensor"""
    data = [
        torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter
    ]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, bsz: int, device: torch.device) -> Tensor:
    """
    Divides the data into bsz seperate sequences, removing extra elements that
    wouldn't cleanly fit.

    Args:
        data: Tensor shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[: seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def get_batch(source: Tensor, i: int, bptt: int = 35) -> tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int (index in train data)

    Returns:
        tuple(data, target) where data has shape [seq_len, batch_size] and
        target has shape[seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target
