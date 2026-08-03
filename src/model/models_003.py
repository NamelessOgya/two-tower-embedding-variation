"""
Plan 003 モデル群
-----------------
3A: ノイズ挿入位置の比較 (input / middle / output)
3B: 多様性損失付き学習アダプタ (BPR + λ * div_loss, 6種)
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.model.models import BaseModel

log = logging.getLogger(__name__)

MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_LENGTH = 128

# ══════════════════════════════════════════════════════════════════════
# 共通ユーティリティ
# ══════════════════════════════════════════════════════════════════════

def _avg_pool_normalize(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    pooled = (last_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=-1)


def load_encoder(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════
# Sub-exp 3A: ノイズ挿入位置
# ══════════════════════════════════════════════════════════════════════

SIGMAS_3A   = [0.01, 0.02, 0.05]
POSITIONS_3A = ["input", "middle", "output"]


def _make_input_hook(sigma: float):
    def hook(module, inp, out):
        return out + torch.randn_like(out) * sigma
    return hook


def _make_middle_hook(sigma: float):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            h = out[0] + torch.randn_like(out[0]) * sigma
            return (h,) + out[1:]
        return out + torch.randn_like(out) * sigma
    return hook


@torch.no_grad()
def precompute_all_noisy_embeddings(
    user_texts: list[str],
    device: torch.device,
    n_trials: int = 5,
    batch_size: int = 256,
    clean_user_embs: np.ndarray | None = None,
) -> dict[tuple[str, float], np.ndarray]:
    """
    mE5 を1回だけロードして全 (position, sigma) の noisy embedding を事前計算。

    最適化:
    - position="output" は clean_user_embs があれば mE5 不要（numpy で加算）
    - position="input"/"middle" は n_trials 回の forward pass が必要

    Returns:
        dict[(position, sigma)] -> (N_users, N_trials, 768) float32
    """
    N = len(user_texts)
    result: dict[tuple, np.ndarray] = {}

    # ── output: clean embeddings + numpy noise (高速) ──────────────
    if clean_user_embs is not None:
        for sigma in SIGMAS_3A:
            log.info(f"  Precomputing position=output, σ={sigma} × {n_trials} trials (numpy) ...")
            all_trials = np.zeros((N, n_trials, 768), dtype=np.float32)
            for trial in range(n_trials):
                noise = np.random.randn(N, 768).astype(np.float32) * sigma
                noisy = clean_user_embs + noise
                norm  = np.linalg.norm(noisy, axis=1, keepdims=True)
                all_trials[:, trial, :] = noisy / np.maximum(norm, 1e-9)
            result[("output", sigma)] = all_trials
            log.info(f"    Done. shape={all_trials.shape}")
    
    # ── input / middle: mE5 forward pass が必要 ────────────────────
    model, tokenizer = load_encoder(device)
    enc = tokenizer(
        user_texts, max_length=MAX_LENGTH, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    for position in [p for p in POSITIONS_3A if p != "output"]:
        for sigma in SIGMAS_3A:
            log.info(f"  Precomputing position={position}, σ={sigma} × {n_trials} trials (mE5) ...")
            all_trials = np.zeros((N, n_trials, 768), dtype=np.float32)

            for trial in range(n_trials):
                handle = None
                if position == "input":
                    handle = model.embeddings.register_forward_hook(_make_input_hook(sigma))
                elif position == "middle":
                    n_layers = len(model.encoder.layer)
                    mid = n_layers // 2  # layer 6 of 12
                    handle = model.encoder.layer[mid].register_forward_hook(
                        _make_middle_hook(sigma)
                    )

                trial_embs = []
                for i in range(0, N, batch_size):
                    ids  = input_ids[i:i+batch_size].to(device)
                    mask = attention_mask[i:i+batch_size].to(device)
                    out  = model(input_ids=ids, attention_mask=mask)
                    pooled = _avg_pool_normalize(out.last_hidden_state, mask)
                    trial_embs.append(pooled.cpu().numpy())

                if handle:
                    handle.remove()
                all_trials[:, trial, :] = np.vstack(trial_embs)

            result[(position, sigma)] = all_trials
            log.info(f"    Done. shape={all_trials.shape}")

    del model
    torch.cuda.empty_cache()
    return result

    del model
    torch.cuda.empty_cache()
    return result


class M_NoisePosition(BaseModel):
    """
    事前計算した noisy embedding を使うモデル。
    trial t → precomputed_embs[user, t % n_trials]
    """

    def __init__(self, position: str, sigma: float, trial_embeddings: np.ndarray):
        self.position = position
        self.sigma = sigma
        self.trial_embeddings = trial_embeddings  # (N_users, N_trials, 768)
        self.n_trials = trial_embeddings.shape[1]
        self.name = f"3A_{position}_s{sigma:.3f}".replace(".", "p")

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        super().prepare(train_pos, user_embeddings, item_embeddings, device)
        # trial_embeddings は既に外で計算済み

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        t = trial % self.n_trials
        return self.trial_embeddings[user_idx, t].copy()


# ══════════════════════════════════════════════════════════════════════
# Sub-exp 3B: 多様性損失付き学習アダプタ
# ══════════════════════════════════════════════════════════════════════

class _DivAdapter(nn.Module):
    """
    ユーザー埋め込みに加える確率的ノイズのスケールを次元ごとに学習する。
    sigma は Softplus で正値に制約。
    """

    def __init__(self, dim: int = 768):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.full((dim,), -4.0))

    def sigma(self) -> torch.Tensor:
        return F.softplus(self.log_sigma)

    def forward(self, user_emb: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(self.log_sigma)
        eps   = torch.randn_like(user_emb)
        return F.normalize(user_emb + sigma * eps, p=2, dim=-1)


# ── 6種の多様性損失関数 ─────────────────────────────────────────────

def div_cosine_emb(q1: torch.Tensor, q2: torch.Tensor,
                   items: torch.Tensor | None = None, T: float = 0.1) -> torch.Tensor:
    """
    [埋め込みレベル] コサイン類似度を最小化 → q1 と q2 の方向を離す。
    Loss = cos_sim(q1, q2)  ← これを最小化
    """
    return (q1 * q2).sum(-1).mean()


def div_l2_emb(q1: torch.Tensor, q2: torch.Tensor,
               items: torch.Tensor | None = None, T: float = 0.1) -> torch.Tensor:
    """
    [埋め込みレベル] L2 距離を最大化 → マイナス L2 を最小化。
    Loss = -||q1 - q2||_2
    """
    return -torch.norm(q1 - q2, dim=-1).mean()


def div_kl_dist(q1: torch.Tensor, q2: torch.Tensor,
                items: torch.Tensor, T: float = 0.1) -> torch.Tensor:
    """
    [スコア分布レベル] KL 乖離度を最大化。
    scores_i = item_embs @ q_i / T → softmax → p_i
    Loss = -KL(p1 || p2)  ← 最小化すると KL が大きくなる
    """
    s1 = (items @ q1.T) / T     # (N_items, B)
    s2 = (items @ q2.T) / T
    log_p1 = F.log_softmax(s1, dim=0)
    p2     = F.softmax(s2, dim=0)
    kl = -(log_p1 * p2).sum(0).mean()   # -KL(p1||p2)
    return kl   # minimize → KL↑（多様性↑）  ※ bugfix: was `return -kl` (double negation)


def div_js_dist(q1: torch.Tensor, q2: torch.Tensor,
                items: torch.Tensor, T: float = 0.1) -> torch.Tensor:
    """
    [スコア分布レベル] Jensen-Shannon 乖離度を最大化。
    対称性があるため KL より安定。
    """
    s1 = (items @ q1.T) / T
    s2 = (items @ q2.T) / T
    p1 = F.softmax(s1, dim=0) + 1e-10
    p2 = F.softmax(s2, dim=0) + 1e-10
    m  = 0.5 * (p1 + p2)
    kl1 = (p1 * (p1 / m).log()).sum(0).mean()
    kl2 = (p2 * (p2 / m).log()).sum(0).mean()
    jsd = 0.5 * (kl1 + kl2)
    return -jsd   # maximize JSD


def div_soft_jaccard(q1: torch.Tensor, q2: torch.Tensor,
                     items: torch.Tensor, T: float = 0.1) -> torch.Tensor:
    """
    [スコア分布レベル] Soft Jaccard 類似度を最小化。
    Soft Jaccard = sum(min(p1,p2)) / sum(max(p1,p2))
    → 推薦リストの「重なり」を直接最小化
    """
    s1 = (items @ q1.T) / T
    s2 = (items @ q2.T) / T
    p1 = F.softmax(s1, dim=0)
    p2 = F.softmax(s2, dim=0)
    inter = torch.min(p1, p2).sum(0).mean()
    union = torch.max(p1, p2).sum(0).mean()
    return inter / (union + 1e-10)   # minimize = maximize diversity


def div_listnet(q1: torch.Tensor, q2: torch.Tensor,
                items: torch.Tensor, T: float = 0.5) -> torch.Tensor:
    """
    [ソフトランクレベル] ListNet スタイル。
    推薦スコア分布のコサイン類似度を最小化 → 順位リストを離す。
    T=0.5 で top-K に集中した分布を使用。
    """
    s1 = (items @ q1.T) / T
    s2 = (items @ q2.T) / T
    p1 = F.softmax(s1, dim=0)
    p2 = F.softmax(s2, dim=0)
    cos = (p1 * p2).sum(0) / (p1.norm(dim=0) * p2.norm(dim=0) + 1e-10)
    return cos.mean()   # minimize rank-list cosine similarity


DIV_LOSSES: dict[str, Callable] = {
    "cosine_emb":   div_cosine_emb,   # 埋め込みレベル
    "l2_emb":       div_l2_emb,       # 埋め込みレベル
    "kl_dist":      div_kl_dist,      # スコア分布レベル
    "js_dist":      div_js_dist,      # スコア分布レベル（対称）
    "soft_jaccard": div_soft_jaccard, # 推薦リストレベル（Jaccard近似）
    "listnet":      div_listnet,      # ソフトランクレベル（ListNet）
}

NEEDS_ITEMS = {"kl_dist", "js_dist", "soft_jaccard", "listnet"}


class M_DiversityAdapter(BaseModel):
    """
    BPR + λ × div_loss で学習する多様性アダプタ。

    訓練:
      q1 = adapter(user_emb)  ← 独立サンプリング
      q2 = adapter(user_emb)
      L  = BPR(q1) + BPR(q2) + λ · div_loss(q1, q2)

    推論:
      query_vec = adapter(user_emb)  ← 試行ごとに異なる noise
    """

    def __init__(
        self,
        div_loss_name: str,
        lambda_div: float = 1.0,
        lr: float = 2e-3,
        epochs: int = 50,
        batch_size: int = 512,
    ):
        self.div_loss_name = div_loss_name
        self.div_fn = DIV_LOSSES[div_loss_name]
        self.lambda_div = lambda_div
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.name = f"3B_{div_loss_name}_l{lambda_div:.1f}".replace(".", "p")
        self.adapter: _DivAdapter | None = None
        self._device: torch.device | None = None

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        super().prepare(train_pos, user_embeddings, item_embeddings, device)
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._train(train_pos, user_embeddings, item_embeddings)

    def _train(self, train_pos, user_embs_np: np.ndarray, item_embs_np: np.ndarray):
        dev = self._device
        X_user = torch.from_numpy(user_embs_np).float().to(dev)
        X_item = torch.from_numpy(item_embs_np).float().to(dev)
        N_items = len(item_embs_np)
        needs_items = self.div_loss_name in NEEDS_ITEMS

        # 訓練ペア構築 (user_idx, pos_item_idx)
        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items[:30]:  # ユーザー当たり最大30ペア
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)
        log.info(f"[{self.name}] Training pairs: {len(pairs_t)}, λ={self.lambda_div}")

        self.adapter = _DivAdapter(dim=768).to(dev)
        opt = torch.optim.Adam(self.adapter.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-5)

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            total, n_b = 0.0, 0

            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx = bp[:, 0]
                p_idx = bp[:, 1]
                n_idx = torch.randint(0, N_items, (len(bp),), device=dev)

                u_emb  = X_user[u_idx]
                p_emb  = X_item[p_idx]
                n_emb  = X_item[n_idx]

                q1 = self.adapter(u_emb)
                q2 = self.adapter(u_emb)   # 独立なサンプリング

                # BPR 損失
                bpr1 = -F.logsigmoid((q1 * p_emb).sum(-1) - (q1 * n_emb).sum(-1)).mean()
                bpr2 = -F.logsigmoid((q2 * p_emb).sum(-1) - (q2 * n_emb).sum(-1)).mean()

                # 多様性損失
                div = self.div_fn(q1, q2, X_item if needs_items else None)

                loss = bpr1 + bpr2 + self.lambda_div * div
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
                opt.step()
                total += loss.item(); n_b += 1

            sched.step()
            if (epoch + 1) % 10 == 0:
                s = self.adapter.sigma()
                log.info(f"[{self.name}] epoch {epoch+1}/{self.epochs}  "
                         f"loss={total/n_b:.4f}  σ_mean={s.mean():.4f}  σ_max={s.max():.4f}")

        self.adapter.eval()
        s = self.adapter.sigma()
        log.info(f"[{self.name}] Done.  σ_mean={s.mean().item():.5f}  σ_max={s.max().item():.5f}")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        u = torch.from_numpy(self.user_embeddings[user_idx]).float()
        u = u.unsqueeze(0).to(self._device)
        with torch.no_grad():
            q = self.adapter(u).squeeze(0)
        return q.cpu().numpy()


def make_diversity_adapter_models(lambda_div: float = 1.0) -> list[M_DiversityAdapter]:
    return [M_DiversityAdapter(name, lambda_div=lambda_div) for name in DIV_LOSSES]
