import os
import copy
import random
from typing import Optional

import ray
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from openfold_tools.openfold_predictor import OpenfoldPredictor
from model.model import SimilarityPredictor
from utils.rewarder import Rewarder
from utils.tools import (
    clean_sequences,
    encode_batch_msa,
    get_all_hypo_embeddings,
    replace_check,
)


# ============================================================
# Dataset
# ============================================================

class ProteinPairDataset(Dataset):
    """
    Each sample:
      pair:      [2, L*21]
      y_rmsd:    scalar
      y_plddt:   [2] (dummy zeros if not available)
      has_plddt: scalar (0/1)
    """
    def __init__(self, embeddings, pair_indices, rmsd_labels, plddt_labels=None):
        """
        embeddings: Tensor [N, L*21]
        pair_indices: list[(i,j)]
        rmsd_labels: list[float]
        plddt_labels: array-like [N] in [0,1] or None
        """
        self.embeddings = embeddings
        self.pair_indices = pair_indices
        self.rmsd_labels = rmsd_labels
        self.plddt_labels = None if plddt_labels is None else np.asarray(plddt_labels, dtype=np.float32)

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        i, j = self.pair_indices[idx]
        emb_i = self.embeddings[i]
        emb_j = self.embeddings[j]
        pair = torch.stack([emb_i, emb_j], dim=0)  # [2, L*21]

        y_rmsd = torch.tensor(self.rmsd_labels[idx], dtype=torch.float32)

        if self.plddt_labels is None:
            y_plddt = torch.zeros((2,), dtype=torch.float32)
            has_plddt = torch.tensor(0.0, dtype=torch.float32)
        else:
            y_plddt = torch.tensor([self.plddt_labels[i], self.plddt_labels[j]], dtype=torch.float32)  # [2]
            has_plddt = torch.tensor(1.0, dtype=torch.float32)

        return pair, y_rmsd, y_plddt, has_plddt


# ============================================================
# Pair sampling
# ============================================================

def sample_balanced_pairs(
    distance_matrix,
    num_pairs=2000,
    seed=42,
    num_bins=10,
    test_ratio=0.4,
    symmetric=True,
):
    """
    Stratified sampling over distance bins.

    - Build unique undirected pairs (i<j)
    - Bin by distance
    - Sample ~equal count each bin
    - Top up if short
    - Optionally add symmetric direction (j,i)
    """
    if isinstance(distance_matrix, torch.Tensor):
        dist = distance_matrix.detach().cpu().numpy()
    else:
        dist = np.asarray(distance_matrix)

    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"distance_matrix must be square, got {dist.shape}")

    rng = np.random.default_rng(seed)
    N = dist.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 items to sample pairs.")

    iu = np.triu_indices(N, k=1)
    dvals = dist[iu].astype(np.float32)

    valid = np.isfinite(dvals)
    i0 = iu[0][valid]
    j0 = iu[1][valid]
    dvals = dvals[valid]
    if dvals.size == 0:
        raise RuntimeError("No finite distances found in distance_matrix.")

    dmin = float(dvals.min())
    dmax = float(dvals.max())

    if dmax <= dmin + 1e-12:
        all_idx = np.arange(dvals.size)
        chosen = rng.choice(all_idx, size=min(num_pairs, all_idx.size), replace=(all_idx.size < num_pairs))
    else:
        num_bins_eff = int(max(2, min(num_bins, num_pairs)))
        edges = np.linspace(dmin, dmax, num_bins_eff + 1, dtype=np.float32)
        bin_ids = np.digitize(dvals, edges[1:-1], right=False)

        per_bin = max(num_pairs // num_bins_eff, 1)
        chosen_list = []
        for b in range(num_bins_eff):
            idxs = np.where(bin_ids == b)[0]
            if idxs.size == 0:
                continue
            k = min(per_bin, idxs.size)
            sel = rng.choice(idxs, size=k, replace=False)
            chosen_list.append(sel)

        chosen = np.concatenate(chosen_list, axis=0) if len(chosen_list) > 0 else np.array([], dtype=int)

        if chosen.size < num_pairs:
            need = num_pairs - chosen.size
            all_idx = np.arange(dvals.size)

            mask = np.ones(dvals.size, dtype=bool)
            mask[chosen] = False
            pool = all_idx[mask]

            if pool.size >= need:
                extra = rng.choice(pool, size=need, replace=False)
            else:
                extra = pool
                if extra.size < need:
                    extra2 = rng.choice(all_idx, size=need - extra.size, replace=True)
                    extra = np.concatenate([extra, extra2], axis=0)

            chosen = np.concatenate([chosen, extra], axis=0)

        chosen = chosen[:num_pairs]

    pairs = list(zip(i0[chosen].tolist(), j0[chosen].tolist()))
    labels = dvals[chosen].astype(np.float32).tolist()

    if symmetric:
        pairs_sym = [(j, i) for (i, j) in pairs]
        labels_sym = labels.copy()
        pairs = pairs + pairs_sym
        labels = labels + labels_sym

    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]
    labels = [labels[i] for i in perm]

    split_idx = int((1.0 - float(test_ratio)) * len(pairs))
    split_idx = max(1, min(split_idx, len(pairs) - 1))

    train_pairs = pairs[:split_idx]
    train_labels = labels[:split_idx]
    test_pairs = pairs[split_idx:]
    test_labels = labels[split_idx:]

    return train_pairs, train_labels, test_pairs, test_labels


# ============================================================
# Train / Eval (supports optional pLDDT loss)
# ============================================================

def _plddt_mse_with_mask(pred_plddt, y_plddt, has_plddt):
    """
    pred_plddt: [B,2]
    y_plddt:    [B,2]
    has_plddt:  [B] 0/1
    """
    mask = has_plddt.view(-1, 1)  # [B,1]
    denom = (mask.sum() * 2.0).clamp_min(1.0)
    return (((pred_plddt - y_plddt) ** 2) * mask).sum() / denom


def evaluate(model, dataset, batch_size=256, device="cuda:0", w_plddt=1.0):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total = 0
    sum_loss = 0.0
    sum_rmsd = 0.0
    sum_plddt = 0.0

    with torch.no_grad():
        for pairs, y_rmsd, y_plddt, has_plddt in loader:
            pairs = pairs.to(device)
            y_rmsd = y_rmsd.to(device)
            y_plddt = y_plddt.to(device)
            has_plddt = has_plddt.to(device)

            pred_rmsd, pred_plddt = model(pairs)

            loss_rmsd = F.mse_loss(pred_rmsd, y_rmsd)

            if has_plddt.sum().item() > 0:
                loss_plddt = _plddt_mse_with_mask(pred_plddt, y_plddt, has_plddt)
                loss = loss_rmsd + float(w_plddt) * loss_plddt
            else:
                loss_plddt = torch.tensor(0.0, device=device)
                loss = loss_rmsd

            bs = pairs.size(0)
            total += bs
            sum_loss += float(loss.item()) * bs
            sum_rmsd += float(loss_rmsd.item()) * bs
            sum_plddt += float(loss_plddt.item()) * bs

    denom = max(total, 1)
    return sum_loss / denom, sum_rmsd / denom, sum_plddt / denom


def train(
    model,
    optimizer,
    train_dataset,
    test_dataset,
    batch_size=256,
    epochs=10,
    num_workers=0,
    device="cuda:0",
    w_plddt=1.0,
):
    model.to(device)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    best = float("inf")
    best_state = None
    rows = []

    for ep in range(epochs):
        model.train()

        total = 0
        sum_loss = 0.0
        sum_rmsd = 0.0
        sum_plddt = 0.0

        for pairs, y_rmsd, y_plddt, has_plddt in loader:
            pairs = pairs.to(device)
            y_rmsd = y_rmsd.to(device)
            y_plddt = y_plddt.to(device)
            has_plddt = has_plddt.to(device)

            pred_rmsd, pred_plddt = model(pairs)
            loss_rmsd = F.mse_loss(pred_rmsd, y_rmsd)

            if has_plddt.sum().item() > 0:
                loss_plddt = _plddt_mse_with_mask(pred_plddt, y_plddt, has_plddt)
                loss = loss_rmsd + float(w_plddt) * loss_plddt
            else:
                loss_plddt = torch.tensor(0.0, device=device)
                loss = loss_rmsd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = pairs.size(0)
            total += bs
            sum_loss += float(loss.item()) * bs
            sum_rmsd += float(loss_rmsd.item()) * bs
            sum_plddt += float(loss_plddt.item()) * bs

        tr_loss = sum_loss / max(total, 1)
        tr_rmsd = sum_rmsd / max(total, 1)
        tr_plddt = sum_plddt / max(total, 1)

        te_loss, te_rmsd, te_plddt = evaluate(
            model, test_dataset,
            batch_size=batch_size,
            device=device,
            w_plddt=w_plddt
        )

        if te_loss < best:
            best = te_loss
            best_state = copy.deepcopy(model.state_dict())

        rows.append({
            "epoch": int(ep + 1),
            "train_loss": float(round(tr_loss, 6)),
            "train_rmsd_mse": float(round(tr_rmsd, 6)),
            "train_plddt_mse": float(round(tr_plddt, 6)),
            "valid_loss": float(round(te_loss, 6)),
            "valid_rmsd_mse": float(round(te_rmsd, 6)),
            "valid_plddt_mse": float(round(te_plddt, 6)),
        })

    return best_state, pd.DataFrame(rows)


# ============================================================
# Memory
# ============================================================

class Memory:
    def __init__(self):
        self.embeddings = []
        self.structures = []
        self.plddts = []  # scalar in [0,1]

    def clear(self):
        del self.embeddings[:]
        del self.structures[:]
        del self.plddts[:]

    def truncate(self, threshold: int) -> list:
        n = len(self.embeddings)
        assert len(self.structures) == n, "Memory: embeddings/structures length mismatch"
        assert len(self.plddts) == n, "Memory: embeddings/plddts length mismatch"

        if n <= threshold:
            return list(range(n))

        selected = random.sample(range(n), threshold)
        self.embeddings = [self.embeddings[i] for i in selected]
        self.structures = [self.structures[i] for i in selected]
        self.plddts = [self.plddts[i] for i in selected]
        return selected


# ============================================================
# Player (CPU inference; GPU train only on trainer0)
# ============================================================

class Player:
    def __init__(
        self,
        res_num: int,
        v: int,
        rmsd_threshold: float,
        plddt_threshold: float,
        infer_device: str = "cpu",
        train_device: str = "cuda:0",
        enable_gpu_train: bool = False,
        lr: float = 1e-4,
        w_plddt: float = 1.0,
    ):
        self.v = int(v)
        self.rmsd_threshold = float(rmsd_threshold)
        self.plddt_threshold = float(plddt_threshold)

        self.infer_device = str(infer_device)
        self.train_device = str(train_device)
        self.enable_gpu_train = bool(enable_gpu_train)

        self.w_plddt = float(w_plddt)

        # CPU model for inference
        self.model_cpu = SimilarityPredictor(res_num=res_num).to("cpu")
        self.model_cpu.eval()

        # GPU model for training (trainer0 only)
        self.model_gpu = None
        self.optimizer = None
        if self.enable_gpu_train:
            if torch.cuda.is_available():
                self.model_gpu = SimilarityPredictor(res_num=res_num).to(self.train_device)
                self.optimizer = optim.Adam(self.model_gpu.parameters(), lr=float(lr))
            else:
                self.model_gpu = SimilarityPredictor(res_num=res_num).to("cpu")
                self.optimizer = optim.Adam(self.model_gpu.parameters(), lr=float(lr))
                self.train_device = "cpu"

        self.memory = Memory()

    def get_params_cpu(self):
        return self.model_cpu.state_dict()

    def set_params(self, params):
        self.model_cpu.load_state_dict(params, strict=True)
        self.model_cpu.eval()
        if self.model_gpu is not None:
            self.model_gpu.load_state_dict(params, strict=True)

    @torch.no_grad()
    def _embed_cpu(self, emb_2d: torch.Tensor) -> torch.Tensor:
        """
        emb_2d: [B, L*21] on CPU
        return: [B, hidden] on CPU
        """
        return self.model_cpu.embed(emb_2d.to("cpu"))

    @torch.no_grad()
    def _rmsd_head_cpu(self, hA: torch.Tensor, hB: torch.Tensor) -> torch.Tensor:
        """
        hA: [B, hidden]
        hB: [B, hidden]
        return: [B] predicted rmsd
        """
        delta = torch.abs(hA - hB)
        combined = torch.cat([delta, hA + hB, hA * hB], dim=-1)  # [B, 3*hidden]
        out = self.model_cpu.rmsd_head(combined).squeeze(-1)
        return out

    def select_action_search(self, state, buffer_emb: torch.Tensor, hypotheticals_emb: torch.Tensor, temp: float):
        """
        search mode: maximize min-distance to buffer
        CPU inference only.
        """
        selected = np.flatnonzero(state["state"] > 0).astype(np.int64)
        unselected = np.flatnonzero(state["state"] <= 0).astype(np.int64)
        if unselected.size == 0:
            raise RuntimeError("No available actions left.")

        v = int(hypotheticals_emb.size(0))
        if v != self.v:
            raise RuntimeError(f"hypotheticals_emb rows={v} but v={self.v}")

        # embed hypos + plddt
        h_hypo = self._embed_cpu(hypotheticals_emb)                 # [v, hidden]
        plddt_pred = self.model_cpu.plddt_head(h_hypo).squeeze(-1)  # [v] in [0,1]

        # embed buffer once
        n = int(buffer_emb.size(0))
        h_buf = self._embed_cpu(buffer_emb)                         # [n, hidden]

        # compute novelty = min predicted rmsd to any buffer item
        novelty = torch.full((v,), float("inf"), dtype=torch.float32)
        for i in range(n):
            hB = h_buf[i].view(1, -1).expand(v, -1)                 # [v, hidden]
            pred = self._rmsd_head_cpu(h_hypo, hB)                  # [v]
            novelty = torch.minimum(novelty, pred)

        logits = novelty.clone()

        # pLDDT hard mask + already-selected mask
        valid_mask = (plddt_pred >= float(self.plddt_threshold))
        logits[~valid_mask] = -1e9
        if selected.size > 0:
            logits[selected] = -1e9

        logits = logits / float(temp)
        probs = F.softmax(logits, dim=0)

        if (not torch.isfinite(probs).all()) or probs.sum().item() <= 0:
            return int(np.random.choice(unselected))

        action = torch.multinomial(probs, 1).item()
        if state["state"][action] > 0:
            action = int(np.random.choice(unselected))
        return int(action)

    def select_action_update(self, state, anchor_emb: torch.Tensor, hypotheticals_emb: torch.Tensor, temp: float):
        """
        update mode: chase similarity to anchor (min predicted rmsd)
        CPU inference only.
        """
        selected = np.flatnonzero(state["state"] > 0).astype(np.int64)
        unselected = np.flatnonzero(state["state"] <= 0).astype(np.int64)
        if unselected.size == 0:
            raise RuntimeError("No available actions left.")

        v = int(hypotheticals_emb.size(0))
        if v != self.v:
            raise RuntimeError(f"hypotheticals_emb rows={v} but v={self.v}")

        h_hypo = self._embed_cpu(hypotheticals_emb)                 # [v, hidden]
        plddt_pred = self.model_cpu.plddt_head(h_hypo).squeeze(-1)  # [v]

        h_anchor = self._embed_cpu(anchor_emb.view(1, -1)).view(1, -1)  # [1, hidden]
        hA = h_anchor.expand(v, -1)                                     # [v, hidden]

        dist = self._rmsd_head_cpu(h_hypo, hA)                          # [v]

        # smaller dist => better, so logits = -dist
        logits = -dist.clone()

        valid_mask = (plddt_pred >= float(self.plddt_threshold))
        logits[~valid_mask] = -1e9
        if selected.size > 0:
            logits[selected] = -1e9

        logits = logits / float(temp)
        probs = F.softmax(logits, dim=0)

        if (not torch.isfinite(probs).all()) or probs.sum().item() <= 0:
            return int(np.random.choice(unselected))

        action = torch.multinomial(probs, 1).item()
        if state["state"][action] > 0:
            action = int(np.random.choice(unselected))
        return int(action)
    
    def select_action_update_local_plddt(
        self,
        state,
        anchor_emb: torch.Tensor,
        hypotheticals_emb: torch.Tensor,
        temp: float,
        d_update: float,
        eta: float = 6.0,
        eps: float = 1e-6,
    ):
        """
        update mode: chase similarity to anchor
        CPU inference only.
        """
        selected = np.flatnonzero(state["state"] > 0).astype(np.int64)
        unselected = np.flatnonzero(state["state"] <= 0).astype(np.int64)
        if unselected.size == 0:
            raise RuntimeError("No available actions left.")

        v = int(hypotheticals_emb.size(0))
        if v != self.v:
            raise RuntimeError(f"hypotheticals_emb rows={v} but v={self.v}")

        # embed hypos -> plddt_pred
        h_hypo = self._embed_cpu(hypotheticals_emb)                 # [v, hidden]
        p = self.model_cpu.plddt_head(h_hypo).squeeze(-1)           # [v] in [0,1]

        # logits = logit(p) * eta
        p = p.clamp(eps, 1.0 - eps)
        logits = (torch.log(p) - torch.log(1.0 - p)) * float(eta)   # [v]

        # soft penalty if dist_pred > d_update
        h_anchor = self._embed_cpu(anchor_emb.view(1, -1)).view(1, -1)  # [1, hidden]
        dist_pred = self._rmsd_head_cpu(h_hypo, h_anchor.expand(v, -1))  # [v]

        scale = max(float(d_update) * 0.5, 1e-6)
        logits = logits - torch.relu(dist_pred - float(d_update)) / scale

        # hard mask
        if selected.size > 0:
            logits[selected] = -1e9

        # softmax sampling
        logits = logits / float(temp)
        probs = F.softmax(logits, dim=0)

        if (not torch.isfinite(probs).all()) or probs.sum().item() <= 0:
            return int(np.random.choice(unselected))

        action = int(torch.multinomial(probs, 1).item())
        if state["state"][action] > 0:
            action = int(np.random.choice(unselected))
        return int(action)
    
    def select_action_success_prob_search(
        self,
        state,
        buffer_emb: torch.Tensor,          # [n, L*21] CPU
        hypotheticals_emb: torch.Tensor,   # [v, L*21] CPU
        temp: float,
        p_thr: float,
        rmsd_thr: float,
        s_p: float = 0.05,                 # pLDDT soft width
        s_d: Optional[float] = None,       # RMSD soft width; None -> auto from rmsd_thr
        eta: float = 4.0,                  # sharpen (z-scored logP * eta)
        eps: float = 1e-8,
        min_std: float = 1e-3,
    ) -> int:
        """
        Search mode: sample an unselected action from a temperature-scaled softmax over z-scored log P_succ, 
        where P_succ = sigmoid((p_pred-p_thr)/s_p) * sigmoid((novelty-rmsd_thr)/s_d) and novelty is the min predicted RMSD to the buffer.
        """
        selected = np.flatnonzero(state["state"] > 0).astype(np.int64)
        unselected = np.flatnonzero(state["state"] <= 0).astype(np.int64)
        if unselected.size == 0:
            raise RuntimeError("No available actions left.")

        v = int(hypotheticals_emb.size(0))
        if v != self.v:
            raise RuntimeError(f"hypotheticals_emb rows={v} but v={self.v}")

        # widths
        s_p = float(max(s_p, 1e-4))
        if s_d is None:
            s_d = max(0.1, 0.2 * float(rmsd_thr))  # auto scale
        s_d = float(max(float(s_d), 1e-4))

        with torch.no_grad():
            # 1) embed hypos, predict pLDDT
            h_hypo = self._embed_cpu(hypotheticals_emb)            # [v, hidden]
            p_pred = self.model_cpu.plddt_head(h_hypo).squeeze(-1) # [v] in [0,1]

            # 2) novelty = min predicted RMSD to buffer
            n = int(buffer_emb.size(0))
            if n <= 0:
                wp = torch.sigmoid((p_pred - float(p_thr)) / s_p)
                logits = torch.log(wp + eps)
            else:
                h_buf = self._embed_cpu(buffer_emb)                # [n, hidden]
                novelty = torch.full((v,), float("inf"), dtype=torch.float32)
                for i in range(n):
                    hB = h_buf[i].view(1, -1).expand(v, -1)
                    pred = self._rmsd_head_cpu(h_hypo, hB)          # [v]
                    novelty = torch.minimum(novelty, pred)

                wp = torch.sigmoid((p_pred - float(p_thr)) / s_p)           # [v]
                wd = torch.sigmoid((novelty - float(rmsd_thr)) / s_d)       # [v]
                ps = wp * wd                                               # [v]
                logits = torch.log(ps + eps)                                # log P_succ


        # mask already selected
        if selected.size > 0:
            logits[selected] = -1e9

        valid_idx = torch.where(logits > -1e8)[0]
        if valid_idx.numel() == 0:
            return int(np.random.choice(unselected))

        mu = logits[valid_idx].mean()
        sd = logits[valid_idx].std().clamp_min(min_std)
        logits = (logits - mu) / sd

        logits = logits * float(eta)
        logits = logits / float(max(temp, 1e-6))

        probs = F.softmax(logits, dim=0)
        
        if (not torch.isfinite(probs).all()) or probs.sum().item() <= 0:
            return int(np.random.choice(unselected))

        action = int(torch.multinomial(probs, 1).item())
        if state["state"][action] > 0:
            action = int(np.random.choice(unselected))
        return action

    def update(self, current_episode: int, batch_size=256, epochs=60, num_workers=0):
        """
        GPU training only (trainer0). After training, sync weights back to CPU model.
        """
        if not self.enable_gpu_train or self.model_gpu is None or self.optimizer is None:
            raise RuntimeError("This Player is not configured for GPU training (enable_gpu_train=False).")

        num_memory = len(self.memory.embeddings)
        if num_memory < 2:
            return self.get_params_cpu(), pd.DataFrame([])

        # RMSD matrix on CPU
        rewarder = Rewarder()
        xyz = np.stack(self.memory.structures)  # [N, L, 3]
        distance_matrix = np.zeros((num_memory, num_memory), dtype=np.float32)
        for i in range(num_memory):
            d = rewarder.calculate(xyz[i], xyz)
            if isinstance(d, torch.Tensor):
                distance_matrix[i] = d.detach().cpu().numpy().astype(np.float32)
            else:
                distance_matrix[i] = np.asarray(d, dtype=np.float32)

        num_distance = num_memory * (num_memory - 1)
        train_pairs, train_labels, test_pairs, test_labels = sample_balanced_pairs(
            distance_matrix,
            num_pairs=min(int(num_distance), 40000),
            seed=42 + int(current_episode),
            num_bins=10,
            test_ratio=0.4,
            symmetric=True,
        )

        embedding_tensor = torch.tensor(np.stack(self.memory.embeddings), dtype=torch.float32)  # CPU
        plddt_labels = np.asarray(self.memory.plddts, dtype=np.float32)                         # [N]

        train_dataset = ProteinPairDataset(embedding_tensor, train_pairs, train_labels, plddt_labels=plddt_labels)
        test_dataset  = ProteinPairDataset(embedding_tensor, test_pairs,  test_labels,  plddt_labels=plddt_labels)

        # sync GPU weights from CPU before training
        self.model_gpu.load_state_dict(self.model_cpu.state_dict(), strict=True)

        device = self.train_device
        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.empty_cache()

        best_state, loss_df = train(
            self.model_gpu,
            self.optimizer,
            train_dataset,
            test_dataset,
            batch_size=int(batch_size),
            epochs=int(epochs),
            num_workers=int(num_workers),
            device=device,
            w_plddt=float(self.w_plddt),
        )

        # sync back to CPU inference model
        self.model_cpu.load_state_dict({k: v.detach().cpu() for k, v in best_state.items()}, strict=True)
        self.model_cpu.eval()

        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.empty_cache()

        return self.get_params_cpu(), loss_df


# ============================================================
# Env
# ============================================================

class Env:
    def __init__(self, fasta, max_seq=128, rmsd_threshold=2.0, plddt_threshold=0.7):
        self.fasta = fasta
        self.rewarder = Rewarder()
        self.max_seq = int(max_seq)
        self.rmsd_threshold = float(rmsd_threshold)
        self.plddt_threshold = float(plddt_threshold)

        self.depth = 0
        self.t = 0
        self.success = False
        self.msa = copy.copy(self.fasta)

    def reset(self, state, buffer, availables):
        self.state = pd.Series(
            {k: (v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v)) for k, v in state.items()},
            dtype=state.dtype,
        )
        if "state" in self.state and isinstance(self.state["state"], np.ndarray):
            self.state["state"].flags.writeable = True

        self.buffer = buffer
        self.availables = availables
        self.depth = 0
        self.t = 0
        self.success = False
        self.msa = copy.copy(self.fasta)

    def get_msa(self):
        msa = copy.copy(self.fasta)
        indices = np.argwhere(self.state["state"]).flatten()
        if len(indices) > 0:
            selected_group = [self.availables.loc[idx.item()] for idx in indices]
            for group in selected_group:
                msa += group
        self.depth = len(msa) // 2 - 1
        self.msa = msa

    def get_reward(self, openfold_predictor: OpenfoldPredictor):
        structure, plddt = openfold_predictor.rl_inference("".join(self.msa), self.fasta[1].strip())
        p = (plddt / 100.0).item()

        self.state["structure"] = structure.detach().cpu().numpy()
        self.state["plddt"] = float(p)

        if len(self.buffer) == 0:
            if p >= self.plddt_threshold:
                self.success = True
            return structure, None

        scores = self.rewarder.calculate(structure, np.stack(self.buffer["structure"]))
        score = float(scores.min().item())
        if p < self.plddt_threshold:
            score = 0.0

        if score > self.rmsd_threshold:
            self.success = True
        return structure, scores

    def step(self, action: int):
        new_state = {k: (v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v)) for k, v in self.state.items()}
        if "state" in new_state and isinstance(new_state["state"], np.ndarray):
            new_state["state"].flags.writeable = True

        self.state = pd.Series(new_state, dtype=self.state.dtype)
        self.state["state"][int(action)] = 1

        self.t += 1
        self.get_msa()

    def done(self):
        if self.success:
            return True
        if self.depth >= self.max_seq:
            return True
        return False


# ============================================================
# Trainer (Ray actor)
# ============================================================

@ray.remote
class Trainer:
    def __init__(
        self,
        input_fasta,
        res_num,
        v,
        name=None,
        number=None,
        max_seq=128,
        rmsd_threshold=2.0,
        plddt_threshold=0.7,
        openfold_device="auto",
        player_device="cpu",
        results_dir=None,
        min_depth_for_value=16,
        enable_gpu_train: Optional[bool] = None,   # default True for trainer0
        train_device: str = "cuda:0",
        w_plddt: float = 1.0,
        offload_openfold_during_update: bool = True,
    ):
        self.env = Env(
            fasta=input_fasta,
            max_seq=max_seq,
            rmsd_threshold=rmsd_threshold,
            plddt_threshold=plddt_threshold,
        )

        self.name = name
        self.number = int(number) if number is not None else -1
        self.res_num = int(res_num)
        self.v = int(v)
        self.count = 0
        self.rmsd_threshold = float(rmsd_threshold)
        self.plddt_threshold = float(plddt_threshold)
        self.min_depth_for_value = int(min_depth_for_value)

        if enable_gpu_train is None:
            enable_gpu_train = (self.number == 0)

        self.player = Player(
            res_num=self.res_num,
            v=self.v,
            rmsd_threshold=self.rmsd_threshold,
            plddt_threshold=self.plddt_threshold,
            infer_device="cpu",
            train_device=str(train_device),
            enable_gpu_train=bool(enable_gpu_train),
            w_plddt=float(w_plddt),
        )

        # OpenFold init (GPU)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            if openfold_device == "auto":
                openfold_device = "cuda:0"
        else:
            openfold_device = "cpu"

        self.openfold_device = str(openfold_device)
        self.alphafold2 = OpenfoldPredictor(device=self.openfold_device)
        self.offload_openfold_during_update = bool(offload_openfold_during_update)

        if results_dir is None:
            if self.name is None:
                raise ValueError("results_dir is None but name is None.")
            results_dir = os.path.join("./results/02_conformation_search", str(self.name))

        self.results_dir = str(results_dir)
        self.result_dir = os.path.join(self.results_dir, "result")
        os.makedirs(self.result_dir, exist_ok=True)

    def reset_env(self, state, buffer, availables):
        self.env.reset(state=state, buffer=buffer, availables=availables)

    def get_env(self):
        return self.env

    def get_memory(self):
        return self.player.memory

    def set_memory(self, memory: Memory):
        self.player.memory = memory

    def get_params(self):
        return self.player.get_params_cpu()

    def set_params(self, params):
        self.player.set_params(params)

    def clear_memory(self):
        self.player.memory.clear()

    def _anchor_from_prev_scores(self, prev_scores):
        if prev_scores is None:
            return None
        try:
            if isinstance(prev_scores, torch.Tensor):
                arr = prev_scores.detach().cpu().numpy().reshape(-1)
            else:
                arr = np.asarray(prev_scores, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return None
            return int(arr.argmin())
        except Exception:
            return None

    def _try_offload_openfold_to_cpu(self):
        if not self.offload_openfold_during_update:
            return False
        moved = False
        try:
            m = getattr(self.alphafold2, "model", None)
            if m is not None and hasattr(m, "cpu"):
                m.cpu()
                moved = True
        except Exception:
            moved = False
        if moved and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return moved

    def _try_restore_openfold_to_gpu(self):
        try:
            m = getattr(self.alphafold2, "model", None)
            if m is not None and hasattr(m, "to") and "cuda" in self.openfold_device:
                m.to(self.openfold_device)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update(self, current_episode, batch_size, epochs, num_workers):
        moved = self._try_offload_openfold_to_cpu()
        try:
            new_params, loss_df = self.player.update(
                current_episode=int(current_episode),
                batch_size=int(batch_size),
                epochs=int(epochs),
                num_workers=int(num_workers),
            )
        finally:
            if moved:
                self._try_restore_openfold_to_gpu()
        return new_params, loss_df

    def train(self, temp=1.0, episode_idx: int = 0, is_update: bool = False):
        """
        is_update=False -> search mode
        is_update=True  -> update mode
        """

        # ---------- if update: pick anchor from buffer and reset env.state to it ----------
        anchor_buf_index = None
        anchor_plddt = None
        anchor_struct = None
        anchor_emb = None

        if is_update:
            buf = self.env.buffer
            if len(buf) > 0:
                eligible = []
                eligible_plddt = []

                
                def _depth_from_statevec(state_vec: np.ndarray) -> int:
                    idxs = np.flatnonzero(state_vec > 0).astype(np.int64)
                    depth = 0
                    for ai in idxs:
                        group = self.env.availables.loc[int(ai)]
                        depth += (len(group) // 2)
                        if depth >= int(self.env.max_seq):
                            break
                    return int(depth)

                for bi in range(len(buf)):
                    row = buf.iloc[bi]
                    st = row.get("state", None)
                    emb = row.get("embedding", None)
                    p0 = row.get("plddt", None)
                    if st is None or emb is None or p0 is None:
                        continue
                    if np.sum(np.asarray(st) > 0) >= int(self.v):
                        continue
                    
                    if _depth_from_statevec(np.asarray(st)) >= int(self.env.max_seq):
                        continue
                    eligible.append(int(bi))
                    eligible_plddt.append(float(p0))

                if len(eligible) > 0:
                    p = np.asarray(eligible_plddt, dtype=np.float64)
                    beta = 5.0
                    w = np.exp(-beta * (p - p.min()))
                    w = w / max(w.sum(), 1e-12)
                    mix = 0.10
                    w = (1.0 - mix) * w + mix * (1.0 / len(w))

                    anchor_buf_index = int(np.random.choice(np.asarray(eligible), p=w))
                else:
                    anchor_buf_index = int(np.random.randint(len(buf)))

                anchor_row = buf.iloc[int(anchor_buf_index)]
                # reset env to anchor state
                self.env.reset(state=anchor_row, buffer=self.env.buffer, availables=self.env.availables)
                self.env.get_msa()

                anchor_plddt = float(anchor_row["plddt"])
                anchor_struct = anchor_row["structure"]
                anchor_emb = torch.tensor(np.asarray(anchor_row["embedding"], dtype=np.float32), dtype=torch.float32)

        # ---------- init embedding from current env.msa ----------
        seqs = clean_sequences(self.env.msa)
        state_emb = encode_batch_msa([seqs])  # CPU tensor [1, L*21]
        self.env.state["embedding"] = state_emb.numpy()[0]
        
        if is_update and anchor_emb is not None:
            anchor_emb = torch.tensor(np.asarray(self.env.state["embedding"], dtype=np.float32), dtype=torch.float32)
        
        d_update = 0.5 * float(self.rmsd_threshold)
        delta_plddt = 0.01

        prev_scores = None

        while True:
            unselected = np.flatnonzero(self.env.state["state"] <= 0).astype(np.int64)
            if unselected.size == 0:
                return 0

            # ---------- action selection ----------
            if len(self.env.buffer) == 0:
                action = int(np.random.choice(unselected))
            else:
                if not is_update:
                    current_depth = max(len(self.env.msa) // 2 - 1, 0)
                    if current_depth < self.min_depth_for_value:
                        action = int(np.random.choice(unselected))
                    else:
                        hypotheticals_emb = get_all_hypo_embeddings(self.env)  # CPU [v, L*21]
                        buffer_emb = torch.tensor(np.stack(self.env.buffer["embedding"]), dtype=torch.float32)  # CPU [n, L*21]
                        action = self.player.select_action_search(self.env.state, buffer_emb, hypotheticals_emb, temp)
                else:
                    hypotheticals_emb = get_all_hypo_embeddings(self.env)  # CPU [v, L*21]
                    action = self.player.select_action_update_local_plddt(
                        self.env.state,
                        anchor_emb=anchor_emb,
                        hypotheticals_emb=hypotheticals_emb,
                        temp=temp,
                        d_update=d_update,
                    )

            self.env.step(action)

            # update state embedding
            seqs = clean_sequences(self.env.msa)
            state_emb = encode_batch_msa([seqs])
            self.env.state["embedding"] = state_emb.numpy()[0]

            # OpenFold eval
            structure, scores = self.env.get_reward(self.alphafold2)
            prev_scores = scores

            # memory: ONLY keep pLDDT >= threshold
            try:
                if float(self.env.state["plddt"]) >= float(self.env.plddt_threshold):
                    self.player.memory.embeddings.append(state_emb.squeeze().detach().cpu().numpy())
                    self.player.memory.structures.append(structure.detach().cpu().numpy())
                    self.player.memory.plddts.append(float(self.env.state["plddt"]))
            except Exception:
                pass

            # ---------- update-local replace: target the chosen anchor ----------
            if is_update and (anchor_buf_index is not None) and (anchor_struct is not None) and (anchor_plddt is not None):
                try:
                    new_p = float(self.env.state["plddt"])
                    if (new_p >= float(self.env.plddt_threshold)) and (new_p > float(anchor_plddt) + float(delta_plddt)):
                        r = self.env.rewarder.calculate(
                            structure,
                            np.asarray(anchor_struct, dtype=np.float32)[None, ...]
                        )
                        r = float(r.min().item())
                        if r <= float(d_update):
                            self.count += 1
                            msa = self.env.msa
                            out_path = os.path.join(
                                self.result_dir,
                                f"{self.number}_{self.count}_ep{int(episode_idx)}_update_replace.a3m",
                            )
                            with open(out_path, "w") as f:
                                f.writelines(msa)

                            print(
                                f"[update-replace] anchor={int(anchor_buf_index)} "
                                f"old_p={float(anchor_plddt):.4f} new_p={new_p:.4f} "
                                f"r(anchor,new)={r:.4f} (d_update={float(d_update):.4f})"
                            )

                            self.env.buffer = self.env.buffer.copy()
                            self.env.buffer.iloc[int(anchor_buf_index)] = self.env.state
                            return 2, int(anchor_buf_index)
                except Exception:
                    pass

            # ---------- original replace check (search semantics) ----------
            if scores is not None:
                replace = replace_check(scores, self.rmsd_threshold)
            else:
                replace = False

            if replace is not False:
                if float(self.env.buffer.iloc[replace]["plddt"]) < float(self.env.state["plddt"]):
                    self.count += 1
                    msa = self.env.msa

                    out_path = os.path.join(
                        self.result_dir,
                        f"{self.number}_{self.count}_ep{int(episode_idx)}_replace.a3m",
                    )
                    with open(out_path, "w") as f:
                        f.writelines(msa)

                    print(
                        f"buffer replaced, old plddt: {float(self.env.buffer.iloc[replace]['plddt']):.4f}, "
                        f"new plddt: {float(self.env.state['plddt']):.4f}, "
                        f"rmsd diff: {float(scores.min().item()):.4f}"
                    )

                    self.env.buffer = self.env.buffer.copy()
                    self.env.buffer.iloc[replace] = self.env.state
                    return 2, int(replace)

            # ---------- done / success ----------
            if self.env.done():
                if self.env.success:
                    self.count += 1
                    msa = self.env.msa
                    out_path = os.path.join(
                        self.result_dir,
                        f"{self.number}_{self.count}_ep{int(episode_idx)}.a3m",
                    )
                    with open(out_path, "w") as f:
                        f.writelines(msa)

                    self.env.buffer = pd.concat([self.env.buffer, self.env.state.to_frame().T], ignore_index=True)

                    print("num buffer:", len(self.env.buffer))
                    if scores is not None:
                        print("current score:", float(scores.min().item()))
                    return 1
                return 0
