# Plan 008: Two-Tower ベースでの多様性手法スイープ（Diversity Loss 統合学習・ノイズ位置バリアント）

実験日: 2026-08-03 | Dataset: MovieLens 1M | Seeds: 5 | K=10 | N_trials=10

---

## 背景・動機

Plan 007 により、**ZCA whitened mE5 埋め込み + Two-Tower MLP (depth=2, hidden=64) + BPR** が
ベースラインの M0_strong 比 **17.8倍（Hit@10: 0.50% → 8.92%）** を達成した。

Post-Noise（推論時に proj_user へノイズ加算）は recall_cum を 3.6〜5.9 倍に改善したが、
soft_jaccard アダプタ（Plan 006/007C の手法）とほぼ同等の性能であった。

本実験（Plan 008）では、この **強力な Two-Tower ベースの上で**：
1. Plan 003 3B で検討した **6種の diversity loss（BPR + 多様性損失の統合学習）** を移植する
2. TrainNoise のノイズ挿入位置（MLP 入力前 vs 出力後）× 推論時ノイズの有無を比較する

---

## 実験構造

### Sub-exp 8A: 参照ベースライン（Plan 007 結果の再利用）

Plan 007 の結果から参照値として利用（再実行なし）：
- `TwoTower_d2_h64`: Hit@10=8.92%, recall_cum=1.11%（多様性なし）
- `TwoTower_d2_h64_postnoise_s0p05`: Hit@10=7.87%, recall_cum=4.04%（ベースライン多様化手法）

### Sub-exp 8B: Two-Tower + Diversity Loss（Plan 003 3B 相当）

最良 Two-Tower (d2, h64) の上で BPR + diversity loss を統合学習（user_head のみ fine-tune）。

```
学習:
    q1 = user_head(whitened + sigma * eps1),  eps1 ~ N(0, I)
    q2 = user_head(whitened + sigma * eps2),  eps2 ~ N(0, I)  ← 独立サンプル
    L = BPR(q1) + BPR(q2) + lambda * div_loss(q1, q2)

推論: user_head(whitened + sigma * eps)  ← trial ごとに異なる eps
```

6種の diversity loss:

| 損失名 | レベル | 定義 |
|---|---|---|
| `cosine_emb` | 埋め込み | cos_sim(q1, q2) |
| `l2_emb` | 埋め込み | −‖q1−q2‖₂ |
| `kl_dist` | スコア分布 | −KL(p1‖p2) |
| `js_dist` | スコア分布 | −JSD(p1, p2) |
| `soft_jaccard` | リスト | min(p1,p2)/max(p1,p2) |
| `listnet` | ソフトランク | cos_sim(rank_dist1, rank_dist2) |

パラメータ:
- lambda ∈ {0.1, 0.5, 1.0}（6損失 × 3λ = 18条件）
- sigma = 0.05（固定、7D の最良値）
- epochs = 30, lr = 2e-3, batch_size = 512

### Sub-exp 8C: TrainNoise 位置バリアント × 推論時ノイズあり

ノイズ挿入位置（2種）× 推論時ノイズあり のバリアント比較。

| モデル | 学習時ノイズ位置 | 推論時ノイズ | 既存比較 |
|---|---|---|---|
| `TwoTowerTrainNoise` (Plan 007E) | MLP 入力前 | なし | 多様性なし（ov=1.0） |
| **`TT_inputnoise_both`** | MLP 入力前 | MLP 入力前 | 本実験新規 |
| **`TT_outputnoise_both`** | MLP 出力後 | MLP 出力後 | 本実験新規 |
| `TwoTowerPostNoise` (Plan 007D) | なし | MLP 出力後 | ベースライン多様化 |

sigma スイープ: {0.01, 0.02, 0.05, 0.1}（2バリアント × 4σ = 8条件）

### Sub-exp 8D: Pareto Frontier 統合プロット

全手法の `Diversity (1 − temporal_overlap)` vs `recall_cum`, `hit@10` を統合プロット。
ベースライン（PostNoise）に対して Pareto 優位な手法を特定する。

---

## 実験コード

- モデル定義: `src/model/models_008.py`
- ランナー: `src/run_experiment_008.py`
- 出力: `report/plan_008/`

```bash
PYTHONPATH=. python3 src/run_experiment_008.py --subexp all --device cuda
```
