import torch
from torch import Tensor
from torch.utils.data import dataset
from torchtext.vocab import Vocab
from typing import Any, Tuple


""" ========================================================================
========================= WikiText =========================================
======================================================================== """


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


def get_batch(source: Tensor, i: int, bptt: int = 35) -> Tuple[Tensor, Tensor]:
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


""" ========================================================================
========================= Multi30k =========================================
======================================================================== """
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.datasets import multi30k, Multi30k
from torch.utils.data.datapipes.iter.grouping import ShardingFilterIterDataPipe
from torch.utils.data import DataLoader

from typing import Callable, Iterable, List, Tuple

# fix broken links
url_t = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
url_v = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
multi30k.URL["train"] = url_t
multi30k.URL["valid"] = url_v


def tensor_transform(token_ids: List[int], bos_idx: int, eos_idx: int) -> Tensor:
    return torch.cat([Tensor(x) for x in ([bos_idx], token_ids, [eos_idx])])


def seq_transforms(
    tokenizer: Callable, vocab: Callable, bos_idx: int, eos_idx: int
) -> Callable:
    def func(x):
        return tensor_transform(vocab(tokenizer(x)), bos_idx, eos_idx)

    return func


def get_collate_fn_callback(
    src_transform: Callable, tgt_transform: Callable, pad_idx: int
) -> Callable:
    def collate_fn(batch: List[Tuple[str, ...]]) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            src_batch: tensor(torch.int64), shape[B, Ns]
            tgt_batch: tensor(torch.int64), shape[B, Nt]
        """
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_transform(src_sample.rstrip("\n")))
            tgt_batch.append(tgt_transform(tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx)
        return src_batch.long().t(), tgt_batch.long().t()

    return collate_fn


def yield_tokens(
    data_iter: Iterable, language_idx: int, tokenizer: Callable
) -> Iterable[str]:
    """
    language_idx: 0 -> src_lang, 1 -> tgt_lang
    """
    for data_sample in data_iter:
        yield tokenizer(data_sample[language_idx])


def get_vocab(
    train_iter: ShardingFilterIterDataPipe,
    tokenizer: Callable,
    language_idx: int,
    special_symbols: List[str],
    unk_idx: int,
) -> Vocab:
    token_gen = yield_tokens(train_iter, language_idx, tokenizer)
    vocab = build_vocab_from_iterator(token_gen, specials=special_symbols)
    vocab.set_default_index(unk_idx)
    return vocab


class Dataset:
    _src_lang: str = "de"
    _tgt_lang: str = "en"
    _UNK_IDX: int = 0
    _PAD_IDX: int = 1
    _BOS_IDX: int = 2
    _EOS_IDX: int = 3
    _ss: List[str] = ["<unk>", "<pad>", "<bos>", "<eos>"]  # special_symbols

    def __init__(self, data_dir: str = ".data"):
        self.data_dir = data_dir
        self.train_iter = self._init_Multi30k("train")
        self.val_iter = None
        self.src_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
        self.tgt_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

        ui = self._UNK_IDX
        self.src_vocab = get_vocab(self.train_iter, self.src_tokenizer, 0, self._ss, ui)
        self.tgt_vocab = get_vocab(self.train_iter, self.tgt_tokenizer, 1, self._ss, ui)

        self.src_transform = seq_transforms(
            self.src_tokenizer, self.src_vocab, self._BOS_IDX, self._EOS_IDX
        )
        self.tgt_transform = seq_transforms(
            self.tgt_tokenizer, self.tgt_vocab, self._BOS_IDX, self._EOS_IDX
        )

    def _init_Multi30k(self, split: str) -> ShardingFilterIterDataPipe:
        print(f"initializing Multi30k with {split} split ...")
        pair = (self._src_lang, self._tgt_lang)
        return Multi30k(self.data_dir, split=split, language_pair=pair)

    @property
    def src_vocab_size(self) -> int:
        return len(self.src_vocab)

    @property
    def tgt_vocab_size(self) -> int:
        return len(self.tgt_vocab)

    def get_dataloader(self, B: int, val: bool = False) -> DataLoader:
        """
        Returns a torch.utils.data.DataLoader that yields (src_batch, tgt_batch)
        tuples, where:
            src_batch: tensor(torch.int64), shape[B, Ns]
            tgt_batch: tensor(torch.int64), shape[B, Nt]
        """

        collate_fn = get_collate_fn_callback(
            self.src_transform, self.tgt_transform, self._PAD_IDX
        )
        if val:
            if self.val_iter is None:
                self.val_iter = self._init_Multi30k("valid")
            return DataLoader(self.val_iter, batch_size=B, collate_fn=collate_fn)
        return DataLoader(self.train_iter, batch_size=B, collate_fn=collate_fn)


if __name__ == "__main__":
    d = Dataset(".data")
