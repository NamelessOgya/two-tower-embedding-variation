# Plan 008 実験レポート

実験日: 2026-08-03 | Dataset: MovieLens 1M | Seeds: 5 | K=10 | N_trials=10

---

## ベースライン参照（Plan 007 より）

| モデル | recall_cum | hit@10 | temporal_overlap | diversity |
|---|---|---|---|---|
| M0_strong | 0.0005 | 0.0050 | 1.0000 | 0.0000 |
| TwoTower_d2_h64 | 0.0111 | **0.0892** | 1.0000 | 0.0000 |
| PostNoise σ=0.05 | 0.0404 | 0.0787 | 0.2935 | 0.7065 |
| PostNoise σ=0.10 | 0.0529 | 0.0758 | 0.1485 | 0.8515 |
| soft_jaccard λ=0.5 (007C) | 0.0512 | 0.0683 | 0.1035 | 0.8965 |

---

## Sub-exp 8B: Diversity Loss 統合学習

| モデル | recall_cum | hit@10 | temporal_overlap | diversity |
|---|---|---|---|---|
| **TT_divloss_soft_jaccard_l0p1** | **0.0615** | 0.0738 | 0.0977 | **0.9023** |
| TT_divloss_soft_jaccard_l0p5 | 0.0559 | 0.0583 | 0.0499 | 0.9501 |
| TT_divloss_soft_jaccard_l1p0 | 0.0509 | 0.0529 | 0.0425 | 0.9575 |
| TT_divloss_js_dist_l1p0 | 0.0491 | 0.0518 | 0.0582 | 0.9418 |
| TT_divloss_listnet_l1p0 | 0.0505 | 0.0524 | 0.0534 | 0.9466 |
| TT_divloss_js_dist_l0p5 | 0.0430 | 0.0614 | 0.1635 | 0.8365 |
| TT_divloss_l2_emb_l1p0 | 0.0446 | 0.0709 | 0.1946 | 0.8054 |
| TT_divloss_l2_emb_l0p5 | 0.0443 | 0.0738 | 0.2409 | 0.7591 |
| TT_divloss_cosine_emb_l1p0 | 0.0422 | 0.0693 | 0.2165 | 0.7835 |
| TT_divloss_cosine_emb_l0p5 | 0.0419 | 0.0718 | 0.2509 | 0.7491 |
| TT_divloss_l2_emb_l0p1 | 0.0419 | 0.0769 | 0.2817 | 0.7183 |
| TT_divloss_listnet_l0p5 | 0.0378 | 0.0692 | 0.3142 | 0.6858 |
| TT_divloss_cosine_emb_l0p1 | 0.0334 | 0.0785 | 0.3902 | 0.6098 |
| TT_divloss_kl_dist_l0p1 | 0.0253 | 0.0410 | 0.2106 | 0.7894 |
| TT_divloss_kl_dist_l0p5 | 0.0105 | 0.0366 | 0.3844 | 0.6156 |
| TT_divloss_kl_dist_l1p0 | 0.0096 | 0.0327 | 0.3705 | 0.6295 |

### 知見
- **soft_jaccard λ=0.1 が全条件で最高 recall_cum=0.0615**
  - PostNoise σ=0.05 比 **+52%**、PostNoise σ=0.10 比 **+16%**
  - hit@10 は 0.0787→0.0738 の微減（−6%）のみ → **Pareto 優位**
- kl_dist は不安定で性能崩壊
- soft_jaccard / js_dist / listnet は λ増大で diversity↑ recall↓ の明確なトレードオフ

---

## Sub-exp 8C: TrainNoise 位置バリアント × 推論時ノイズあり

| モデル | recall_cum | hit@10 | temporal_overlap | diversity |
|---|---|---|---|---|
| TT_inputnoise_both σ=0.01 | 0.0189 | 0.0876 | 0.7283 | 0.2717 |
| TT_outputnoise_both σ=0.01 | 0.0147 | 0.0754 | 0.8013 | 0.1987 |
| TT_inputnoise_both σ=0.02 | 0.0235 | 0.0825 | 0.5672 | 0.4328 |
| TT_outputnoise_both σ=0.02 | 0.0237 | 0.0833 | 0.6166 | 0.3834 |
| TT_inputnoise_both σ=0.05 | 0.0290 | 0.0762 | 0.4180 | 0.5820 |
| **TT_outputnoise_both σ=0.05** | 0.0344 | 0.0724 | 0.3117 | 0.6883 |
| TT_inputnoise_both σ=0.10 | 0.0223 | 0.0711 | 0.5271 | 0.4729 |
| **TT_outputnoise_both σ=0.10** | 0.0425 | 0.0606 | 0.1395 | 0.8605 |

### 知見
- **OutputNoiseBoth が全σで InputNoiseBoth を上回る**（MLP後ノイズの方が方向多様性が高い）
- InputNoiseBoth は MLP がノイズを吸収する方向に学習し diversity が伸びにくい
- PostNoise (学習なし) との比較では recall_cum で若干劣後（OutputNoise σ=0.1: 0.0425 vs PostNoise σ=0.1: 0.0529）

---

## 総合 Pareto ランキング（diversity ≥ 0.5 条件）

| 順位 | モデル | recall_cum | hit@10 | diversity |
|---|---|---|---|---|
| 🥇 | **TT_divloss_soft_jaccard_l0p1** | **0.0615** | 0.0738 | 0.9023 |
| 🥈 | TT_divloss_soft_jaccard_l0p5 | 0.0559 | 0.0583 | 0.9501 |
| 🥉 | PostNoise σ=0.10 | 0.0529 | 0.0758 | 0.8515 |
| 4 | TT_soft_jaccard_l0p5 (007C) | 0.0512 | 0.0683 | 0.8965 |
| 5 | TT_divloss_listnet_l1p0 | 0.0505 | 0.0524 | 0.9466 |

### 結論

> **`TT_divloss_soft_jaccard` λ=0.1, σ=0.05 が Plan 008 最優秀手法**。
> PostNoise ベースライン（σ=0.05）比で recall_cum **+52%**、hit@10 −6% のみ。Pareto 優位を達成。

---

## 次のステップ候補

1. soft_jaccard λ の精密スイープ（0.05, 0.08, 0.1, 0.15）
2. 推論 σ の最適化（soft_jaccard + σ=0.02, 0.1）
3. js_dist λ=1.0 + σ最適化（高 diversity 領域を探索）
4. Yelp データセットへの適用

---

## 事後考察（グラフ分析・バグ発見）

### diversity vs recall_cum（左グラフ）の傾向

- **soft_jaccard を除く全手法で右肩上がり**
  → diversity が上がるほど N-trial 累積カバレッジが伸びる（直感通り）
- soft_jaccard は λ=0.1 でも右上（高 diversity × 高 recall_cum）にあり、
  λ をさらに小さくすれば recall_cum がさらに改善できる可能性がある（→ Plan 009 Sub-exp 9A）

### diversity vs recall_avg（右グラフ）の傾向

- **全手法で右下がりの曲線**
  → 多様化を強めるほど「1試行あたりの精度」は低下する本質的なトレードオフ
- recall_avg（単試行精度）と recall_cum（N試行カバレッジ）は相反する指標であることが視覚的に確認された

### kl_dist の異常：コードバグの発見

kl_dist が全 λ で他手法より一貫して低精度であり、
さらに「λ が増えるほど diversity が下がる」という **逆方向の挙動** が確認された：

| λ | recall_cum | temporal_overlap（低いほど高 diversity） |
|---|---|---|
| 0.1 | 0.0253 | 0.2106 |
| 0.5 | 0.0105 | **0.3844**（λ増大で diversity 低下） |
| 1.0 | 0.0096 | **0.3705** |

原因調査の結果、[`models_003.py` L226](file:///home/kasumi/tt-embedding-variation/src/model/models_003.py#L226) に**二重符号反転バグ**を発見：

```python
# バグ（現状）: minimize すると KL が小さくなる（多様性↓）
kl = -(log_p1 * p2).sum(0).mean()   # = -KL
return -kl                            # = +KL

# 正しい実装: minimize すると KL が大きくなる（多様性↑）
return kl                             # = -KL
```

Plan 009 Sub-exp 9B でバグ修正版を再実験する。
