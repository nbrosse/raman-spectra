"""
Training and evaluation loops for SpecBERT on Raman spectra.

Supports:
- Pretraining with Masked Spectral Modeling (MSM), reconstructing masked patches with MSE on masked positions only
- Fine-tuning for classification/regression using CLS token
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .data import (
    LabeledSpectraDataset,
    MaskedSpectraDataset,
    combine_instrument_data,
    load_csv,
    load_numpy,
    load_raman_challenge_dataset,
)
from .model import SpecBERT
from .preprocess import SpectralPreprocessor


def collate_msm(batch):
    wns, ys, masks = zip(*batch)
    wn = wns[0]  # all identical post-preprocess
    x = torch.stack([y for y in ys], dim=0)
    mask = torch.stack([m for m in masks], dim=0)
    return wn, x, mask


def collate_supervised(batch):
    wns, ys, labels = zip(*batch)
    wn = wns[0]
    x = torch.stack([y for y in ys], dim=0)
    y = torch.stack([torch.as_tensor(l) for l in labels], dim=0)
    return wn, x, y


def pretrain_msm(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    model: SpecBERT,
    preprocessor: SpectralPreprocessor | None,
    cfg: TrainConfig,
):
    dataset = MaskedSpectraDataset(wavenumbers, spectra, preprocessor, cfg.patch_size, cfg.mask_ratio)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_msm
    )
    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss(reduction="none")

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"pretrain epoch {epoch+1}/{cfg.epochs}")
        running = 0.0
        count = 0
        for _, x, patch_mask in pbar:
            x = x.to(cfg.device)  # (B, N)
            patch_mask = patch_mask.to(cfg.device)  # (B, P)
            optimizer.zero_grad()
            _, recon, _ = model(x)
            assert recon is not None
            # Build targets grouped by patches
            b, n = x.shape
            p = cfg.patch_size
            x_patched = x.view(b, -1, p)
            loss_all = loss_fn(recon, x_patched).mean(dim=-1)  # (B, P)
            masked_loss = (loss_all * patch_mask.float()).sum() / (patch_mask.float().sum() + 1e-8)
            masked_loss.backward()
            optimizer.step()
            running += masked_loss.item() * x.size(0)
            count += x.size(0)
            pbar.set_postfix({"loss": running / max(1, count)})


def finetune_supervised(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    labels: np.ndarray,
    model: SpecBERT,
    preprocessor: SpectralPreprocessor | None,
    cfg: TrainConfig,
):
    dataset = LabeledSpectraDataset(wavenumbers, spectra, preprocessor, labels)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_supervised
    )
    model = model.to(cfg.device)
    head_params = model.classifier.parameters() if model.classifier is not None else model.parameters()
    optimizer = torch.optim.AdamW(head_params, lr=cfg.lr)
    is_classification = labels.ndim == 1 and labels.dtype in (np.int32, np.int64)
    loss_fn = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"finetune epoch {epoch+1}/{cfg.epochs}")
        running = 0.0
        count = 0
        for _, x, y in pbar:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            optimizer.zero_grad()
            _, _, logits = model(x)
            if logits is None:
                raise RuntimeError("Model has no classifier head for supervised fine-tuning")
            if is_classification:
                y = y.long()
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            count += x.size(0)
            pbar.set_postfix({"loss": running / max(1, count)})


@torch.no_grad()
def evaluate_supervised(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    labels: np.ndarray,
    model: SpecBERT,
    preprocessor: SpectralPreprocessor | None,
    cfg: TrainConfig,
):
    dataset = LabeledSpectraDataset(wavenumbers, spectra, preprocessor, labels)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_supervised
    )
    model = model.to(cfg.device)
    is_classification = labels.ndim == 1 and labels.dtype in (np.int32, np.int64)
    loss_fn = nn.CrossEntropyLoss(reduction="sum") if is_classification else nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total = 0
    correct = 0
    model.eval()
    for _, x, y in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        _, _, logits = model(x)
        if logits is None:
            raise RuntimeError("Model has no classifier head for supervised evaluation")
        if is_classification:
            y = y.long()
        loss = loss_fn(logits, y)
        total_loss += loss.item()
        total += x.size(0)
        if is_classification:
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
    avg_loss = total_loss / max(1, total)
    metrics = {"loss": avg_loss}
    if is_classification:
        metrics["accuracy"] = correct / max(1, total)
    return metrics


def main(cfg: TrainConfig):
    """Main training function that accepts a TrainConfig instance.
    
    Parameters
    ----------
    cfg : TrainConfig
        Configuration object containing all training parameters
    """
    # Load data based on format
    if cfg.format == "csv":
        wn, spectra, _ = load_csv(cfg.input_path)
        labels = np.load(cfg.labels_path) if cfg.labels_path else None
    elif cfg.format == "npy":
        wn, spectra = load_numpy(cfg.input_path, cfg.wavenumbers_path)
        labels = np.load(cfg.labels_path) if cfg.labels_path else None
    elif cfg.format == "challenge":
        # Load challenge dataset
        instrument_data, target_df = load_raman_challenge_dataset(cfg.input_path, cfg.instruments)

        # Combine instrument data
        wn, spectra, instrument_labels = combine_instrument_data(
            instrument_data,
            interpolate_wavenumbers=True
        )

        # Extract target labels if available and in finetune mode
        labels = None
        if cfg.mode == "finetune" and target_df is not None:
            if cfg.target_column in target_df.columns:
                # Match target data to spectra (assuming same order)
                # This is a simplification - in practice you'd need proper matching logic
                target_values = target_df[cfg.target_column].dropna().values
                if len(target_values) == len(spectra):
                    labels = target_values
                else:
                    print(f"Warning: Target data length ({len(target_values)}) != spectra length ({len(spectra)})")
                    print("Using first available targets for matching spectra count")
                    min_len = min(len(target_values), len(spectra))
                    labels = target_values[:min_len]
                    spectra = spectra[:min_len]
                    wn = wn  # wavenumbers stay the same
            else:
                raise ValueError(f"Target column '{cfg.target_column}' not found in target data")
    else:
        raise ValueError(f"Unknown format: {cfg.format}")

    preproc = SpectralPreprocessor()

    model = SpecBERT(
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.heads,
        recon_head=True,
        num_classes=cfg.classes if cfg.mode == "finetune" else None,
    )
    
    # Create training config (subset of full config for training loop)
    train_cfg = TrainConfig(
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        epochs=cfg.epochs,
        patch_size=cfg.patch_size,
        mask_ratio=cfg.mask_ratio,
        num_workers=cfg.num_workers,
        device=cfg.device,
    )

    if cfg.mode == "pretrain":
        pretrain_msm(spectra, wn, model, preproc, train_cfg)
        # save encoder weights
        out = Path(cfg.input_path).with_suffix(".specbert.pt")
        torch.save(model.state_dict(), out)
        print(f"Saved checkpoint to {out}")
    else:
        if labels is None:
            raise SystemExit("Labels are required for finetune mode")
        if cfg.val_split and 0.0 < cfg.val_split < 0.5:
            # simple holdout
            num = spectra.shape[0]
            idx = np.arange(num)
            rng = np.random.default_rng(42)
            rng.shuffle(idx)
            split = int(num * (1 - cfg.val_split))
            tr, va = idx[:split], idx[split:]
            finetune_supervised(spectra[tr], wn, labels[tr], model, preproc, train_cfg)
            metrics = evaluate_supervised(spectra[va], wn, labels[va], model, preproc, train_cfg)
            print({k: float(v) for k, v in metrics.items()})
        else:
            finetune_supervised(spectra, wn, labels, model, preproc, train_cfg)


if __name__ == "__main__":
    # Example usage: create a config and run training
    # Users should instantiate TrainConfig with their desired parameters
    from .config import TrainConfig
    
    # This is just an example - users should provide their own config
    example_cfg = TrainConfig(
        mode="pretrain",
        input_path="",
        format="csv",
    )
    
    if example_cfg.input_path:
        main(example_cfg)
    else:
        print("Please configure TrainConfig with your parameters and call main(cfg)")
