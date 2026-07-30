# Plan 003: Gaussian ノイズ挿入位置の比較 / 多様性損失付き学習アダプタ

実験日: 2026-07-30〜31 | Dataset: MovieLens 1M | Seeds: 5 | K=10 | N_trials=5

---

## 背景・動機

Plan 002 でノイズをモデル出力（pooling後）に加える手法 M4 が最も有効であることが判明した。
一方で「ノイズをモデル内部（input/middle）に入れたらどうなるか」という疑問と、
「BPR 訓練でノイズ方向を学習できないか」という仮説を検証する。

---

## Sub-exp 3A: Gaussian ノイズ挿入位置の比較

挿入位置 = {input, middle, output} × σ = {0.01, 0.02, 0.05} = 9 条件

```
input  : tokenize → Token Embedding + ε(σ) → SA×12 → AvgPool → L2Norm → query
middle : tokenize → SA×6 → h + ε(σ) → SA×6 → AvgPool → L2Norm → query
output : tokenize → SA×12 → AvgPool → emb + ε(σ) → L2Norm → query  (= M4相当)
```

**結果の要点**:
- input/middle ノイズは LayerNorm に吸収されてほぼ無効（overlap≒0.92〜0.98）
- output ノイズのみが有効（σ=0.05 で overlap=0.15, coverage=97%）
- 同じ σ で output は input の **8倍** の多様化効果

詳細: `report/plan_003/noise_position/`

---

## Sub-exp 3B: 多様性損失付き学習アダプタ

アダプタ: `σ_vec ∈ ℝ^768` (per-dimension, Softplus正値制約) を BPR + div_loss で訓練

```
L = BPR(q1) + BPR(q2) + λ · div_loss(q1, q2)
  q1, q2 = L2_normalize(user_emb + σ ⊙ ε),  ε ~ N(0,I)
  λ=1.0固定, epochs=50, Adam lr=2e-3
```

6種の div_loss:

| 損失名 | レベル | 定義 |
|---|---|---|
| `cosine_emb` | 埋め込み | cos_sim(q1, q2) |
| `l2_emb` | 埋め込み | −‖q1−q2‖₂ |
| `kl_dist` | スコア分布 | −KL(p1‖p2) |
| `js_dist` | スコア分布 | −JSD(p1,p2) |
| `soft_jaccard` | リスト | min(p1,p2)/max(p1,p2) |
| `listnet` | ソフトランク | cos_sim(rank_dist1, rank_dist2) |

**結果の要点**:
- `cosine_emb` が最高性能: recall_cum=0.0238, overlap=0.003, coverage=100%
- 未学習の M4 gauss σ=0.2 (recall_cum=0.0221) を +7.7% 上回る
- 埋め込みレベル損失 > スコア分布損失 > リスト損失（今回の設定では）
- λ=1.0固定のため λ スイープは Plan 004 で実施

詳細: `report/plan_003/diversity_adapter/`

---

## ファイル

- 実験コード: `src/run_experiment_003.py`, `src/model/models_003.py`
- レポート: `report/plan_003/report_003.md`
- プロット: `report/plan_003/tradeoff_003.png`
