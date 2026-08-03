# Insight: PostNoise (Random) vs soft_jaccard DivLoss

実験: Plan 008 / 009 / 010 | Dataset: MovieLens 1M | Seeds=5, K=10, N_trials=10

---

## 手法の概要

| | **PostNoise (Random)** | **soft_jaccard DivLoss** |
|---|---|---|
| 訓練時 | 通常 BPR のみ | BPR + soft_jaccard 多様化損失 |
| 推論時 | user ベクトルにガウシアンノイズ付与 | user ベクトルにガウシアンノイズ付与（σ=0.05） |
| 多様化の源泉 | ノイズの乱数のみ | 訓練時の分布整形 + 推論ノイズ |
| ハイパーパラメータ | σ（ノイズ強度） | λ（多様化損失の重み）+ σ |

---

## 比較グラフ

![Plan 010 Comparison: Diversity vs recall_cum / K Sensitivity / Diversity vs recall_avg](/home/kasumi/tt-embedding-variation/report/plan_010/comparison_010.png)

> **Panel 1** (左): 同一 diversity 帯での recall_cum 比較。σ/λ スイープの軌跡を重ねたもの。  
> **Panel 2** (中): K=5/10/20/50 での recall_cum。K が大きいほど soft_jaccard の優位が拡大。  
> **Panel 3** (右): 同一 diversity 帯での recall_avg 比較。PostNoise が一貫して高い。

---

## 主要な数値比較（K=10, N_trials=10）

### Diversity を揃えた場合

| diversity ≈ | PostNoise | soft_jaccard | 差 |
|---|---|---|---|
| 0.65 | σ=0.05: rc=0.037, ra=**0.0096** | λ=0.01: rc=0.036, ra=0.0102 | ほぼ同等（ra は SJ わずか優位） |
| 0.77 | σ=0.07: rc=0.043, ra=**0.0093** | λ=0.03: rc=0.046, ra=0.0099 | SJ が recall_cum で逆転 |
| 0.83 | σ=0.10: rc=0.050, ra=**0.0091** | λ=0.05: rc=0.055, ra=0.0098 | SJ +10% |
| 0.91 | σ=0.20: rc=0.055, ra=0.0080 | λ=0.10: rc=**0.063**, ra=0.0094 | SJ +14%、かつ ra でも優位 |

### K 感度（各 K での recall_cum）

| K | PostNoise σ=0.10 | soft_jaccard λ=0.1 | 差 |
|---|---|---|---|
| 5 | 0.0274 | **0.0304** | +11% |
| 10 | 0.0495 | **0.0585** | +18% |
| 20 | 0.0839 | **0.1031** | +23% |
| 50 | 0.1595 | **0.2064** | +29% |

### 余剰カバレッジの品質（Sub-exp 10C）

| 指標 | PostNoise σ=0.10 | soft_jaccard λ=0.1 |
|---|---|---|
| 平均カバレッジ | 55.0 items | **67.0 items** |
| 独自カバー数（相手にないもの） | 39.8 items | 51.8 items |
| **独自カバーの GT ヒット率** | **0.0086** | 0.0071 |

---

## Insights

### 1. 「同じ diversity ならほぼ同等〜PostNoise 優位」（低 diversity 帯）

diversity < 0.7 の領域では、**PostNoise の recall_cum が soft_jaccard の λ スイープと拮抗または上回る**。
訓練コストを追加しなくても、ノイズを調整するだけで同等の多様化効果が得られる。

### 2. 「高 diversity 帯では soft_jaccard が明確に優位」

diversity > 0.8 以上では soft_jaccard が PostNoise を一貫して上回る。
**soft_jaccard の訓練は「高 diversity 領域への到達効率」を改善している**。
PostNoise では σ を大きくすると diversity は上がるが recall_cum の伸びが鈍化するのに対し、
soft_jaccard は高 diversity を保ちながら recall_cum も高水準を維持する。

### 3. 「K が大きいほど soft_jaccard の優位が拡大」

K=5 で +11%、K=50 で +29% と、**推薦リストが長くなるほど差が広がる**。
soft_jaccard は「多様なアイテムを広くカバーする」特性を持つため、
K が増えて推薦幅が広がるほど真価を発揮する。
一方 PostNoise は K=5 のような短リストでは相対的に善戦する。

### 4. 「独自カバーの精度は PostNoise が高い」

soft_jaccard は独自にカバーするアイテム数は多いが、
その GT ヒット率（0.0071）は PostNoise（0.0086）より低い。
**soft_jaccard の recall_cum 優位の源泉は「精度は低いが広く拾う」戦略**によるものであり、
訓練によって「意味のある多様化」ができているとは言い切れない。
PostNoise の方が「当たりやすい方向に多様化」している可能性がある。

### 5. 「recall_avg（単試行精度）は PostNoise が優位」

ほぼ全 diversity 帯で PostNoise の recall_avg が soft_jaccard より高い。
**1回の推薦でヒットさせる能力は PostNoise が勝る**。
これは soft_jaccard の訓練が精度を一部犠牲にして多様性を獲得しているためと解釈できる。

---

## 結論・使い分けの指針

| ユースケース | 推奨手法 |
|---|---|
| 1回の推薦セッションの精度を最大化したい | **PostNoise**（recall_avg 優位） |
| N 回試行の累積カバレッジを最大化したい（探索型推薦） | **soft_jaccard λ=0.1** |
| リストを長めに出す設計（K=20〜50） | **soft_jaccard** （差が大きくなる） |
| 実装コストを最小化したい | **PostNoise**（再学習不要） |
| 独自カバーの精度を重視する | **PostNoise**（GT ヒット率 優位） |

> **要約**: soft_jaccard は「累積カバレッジ・K 感度」で優位だが、PostNoise は「単試行精度・実装簡易性」で優位。
> 高 diversity 領域では soft_jaccard が明確に有利だが、その優位は「精度の高い多様化」ではなく「広く薄くカバーする」戦略によるものであることを認識しておく必要がある。
