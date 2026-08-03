# Plan 010: PostNoise vs soft_jaccard DivLoss の詳細比較分析

実験日: TBD | Dataset: MovieLens 1M | Seeds: 5 | K=10 | N_trials=10

---

## 背景・動機

Plan 008/009 の結果において、cum-recall (X軸: diversity) の観点で
**PostNoise（推論時ノイズのみ）** と **soft_jaccard DivLoss（訓練時 + 推論時ノイズ）** が
類似した挙動を示した。

### Plan 009 グラフから見える構造

| diversity 帯 | PostNoise | soft_jaccard |
|---|---|---|
| 0.58〜0.71 | σ=0.05: rc=0.0404 | λ=0.001〜0.05: rc=0.032〜0.055 |
| 0.85〜0.91 | σ=0.10: rc=0.0529 | λ=0.1: rc=0.0627 |

**観察**：
- 同じ diversity 帯では **PostNoise の方が recall_cum が高い場合がある**（λ 小の軌跡で）
- λ=0.1 以上では soft_jaccard が逆転して PostNoise を上回る
- recall_avg（単試行精度）では soft_jaccard が優位な可能性

**根本的な問い：**
> soft_jaccard の DivLoss 訓練は PostNoise に対して **本質的な差**をもたらしているか？
> それとも「同じ多様化効果を別の手段で実現しているだけ」か？

---

## 実験構造

### Sub-exp 10A: diversity マッチ比較

同一 diversity レベルで両手法を直接比較する。

- PostNoise σ スイープ（細かく）: σ ∈ {0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20}
- soft_jaccard λ スイープ（Plan 009 結果を流用 + 補足）: λ ∈ {0.001, 0.005, 0.01, 0.03, 0.05, 0.1}
- 同一 diversity 水準（diversity ≈ 0.6, 0.7, 0.8, 0.9）での recall_cum / recall_avg を補間比較

**仮説**: 同一 diversity 水準での recall_avg は soft_jaccard > PostNoise が成立するか。

### Sub-exp 10B: K 感度分析

K=5, 10, 20, 50 で両手法の優劣が変わるかを確認。

- モデル: PostNoise σ=0.05, σ=0.10 vs soft_jaccard λ=0.1
- K ∈ {5, 10, 20, 50}
- 指標: recall_cum@K, recall_avg@K, hit@K

**仮説**: K が小さいほど per-trial 精度が重要になり PostNoise 優位？
        K が大きいほど cumulative coverage が支配的になり soft_jaccard 優位？

### Sub-exp 10C: 余剰カバレッジの品質分析

soft_jaccard が PostNoise より多くカバーする「追加アイテム」の質を評価。

- 分析対象: soft_jaccard λ=0.1 vs PostNoise σ=0.10（diversity が近い2点）
- 指標:
  - `extra_precision`: soft_jaccard が独自にカバーするアイテムのうち、
    真の GT に含まれる割合（余剰カバレッジの精度）
  - `extra_coverage`: アイテム種類数の差
  - User-level: 恩恵を受けるユーザー割合

**仮説**: soft_jaccard の追加カバレッジは「意味のある多様化」（GT に含まれる）をより多く含むか？

### Sub-exp 10D: 統計的有意差検定

5 seeds の分布から t 検定で優劣を検証。

- recall_cum @ diversity ≈ 0.85〜0.91 帯での有意差
- 比較対: (PostNoise σ=0.10) vs (soft_jaccard λ=0.1)

---

## 期待される発見

1. **diversity マッチ比較で soft_jaccard が優位** なら、
   DivLoss 訓練が「同じ多様化コストでより質の高い推薦」を実現している証拠
2. **ほぼ同等** なら、soft_jaccard の優位は diversity の高さによるものであり、
   PostNoise をさらに強くすれば追いつける可能性がある
3. **K 依存性** から、ユースケースによる使い分けの指針を得る

---

## 実装計画

- モデルの再学習: Sub-exp 10A のみ PostNoise σ 追加スイープが必要（10B/10C/10D は既存結果を使用）
- 分析スクリプト: `src/run_experiment_010.py`
- 出力: `report/plan_010/`

```bash
PYTHONPATH=. python3 src/run_experiment_010.py --subexp all --device cuda
```
