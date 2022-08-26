import torch
from torch import nn, Tensor
from data_utils import Dataset
from utils import create_masks
from torch.optim import Optimizer

from typing import Callable


def train_epoch(
    d: Dataset,
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    B: int,
    device: torch.device,
    log_interval: int = 0,
    early_stop: int = 0,
) -> int:
    model.train()
    losses = 0
    train_dataloader = d.get_dataloader(B)

    for i, (src, tgt) in enumerate(train_dataloader):
        # (B, Ns), (B, Nt)
        src, tgt = src.to(device), tgt.to(device)

        # (B, Nt-1), (B, Nt-1)
        tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]

        # (Ns, Ns), (Nt-1, Nt-1), (B, Ns), (B, Nt-1)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(
            src, tgt_inp, d._PAD_IDX, device
        )

        # (B, Nt-1, tgt_vocab_size) -> (B * (Nt-1), tgt_vocab_size)
        logits = model(
            src,
            tgt_inp,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        ).reshape(-1, d.tgt_vocab_size)

        optimizer.zero_grad()
        loss = loss_fn(logits, tgt_out.reshape(-1).long())
        losses += loss.item()
        loss.backward()
        optimizer.step()

        if log_interval and i % log_interval == 0 and i > 0:
            print(f"loss: {losses/(i+1):<.3f}")

        if early_stop and i >= early_stop:
            break

    return losses / (i + 1)


def evaluate(
    d: Dataset,
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    B: int,
    device: torch.device,
) -> int:
    model.eval()
    losses = 0
    val_dataloader = d.get_dataloader(B, val=True)

    for i, (src, tgt) in enumerate(val_dataloader):
        # (B, Ns), (B, Nt)
        src, tgt = src.to(device), tgt.to(device)

        # (B, Nt-1)
        tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]

        # (Ns, Ns), (Nt-1, Nt-1), (B, Ns), (B, Nt-1)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(
            src, tgt_inp, d._PAD_IDX, device
        )

        # (B, Nt-1, tgt_vocab_size) -> (B * (Nt-1), tgt_vocab_size)
        logits = model(
            src,
            tgt_inp,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        ).reshape(-1, d.tgt_vocab_size)

        loss = loss_fn(logits, tgt_out.reshape(-1))
        losses += loss.item()

    return losses / (i + 1)
