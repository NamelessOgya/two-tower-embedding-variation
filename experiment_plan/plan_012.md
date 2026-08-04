# Plan 012: Advanced Soft-Jaccard Variants for High-Precision Diversity

## 目的・背景
Plan 008〜011 の検証により、`soft_jaccard DivLoss` (BPR + Soft Jaccard) が累積再現率 (`recall_cum = 0.0627`) において圧倒的最高性能を示すことが実証された。
しかし、soft_jaccard には以下の課題が存在する：
1. **「広く薄く撒き散らす」探索傾向**: 独自カバーアイテムの GT ヒット率が PostNoise より低い (`0.0071` vs `0.0086`)。
2. **単体精度 (recall_avg) の低下**: Softmax 全体に対するペナルティにより、1試行あたりの精度がベースライン比で低下。
3. **均一ガウシアンノイズ依存**: 全次元一律のノイズ付与に依存している。

本 Plan 012 では、これら 3 つの課題を克服する先進的 Soft-Jaccard 拡張モデルを構築・評価する。

---

## 実験デザイン

### 12A: Top-K Truncated Soft Jaccard
- 全アイテムの Softmax ではなく、上位 $K_{loss}=30$ に確率集中させた分布に対して Soft Jaccard 損失を計算。
- 低順位アイテムの分布破壊を抑え、単体精度 (`recall_avg`) の維持を狙う。

### 12B: Learned Per-Dimension Noise (Adaptive Soft-Jaccard)
- 次元別ノイズスケールベクトル $\boldsymbol{\sigma} \in \mathbb{R}^d$ を定義し、Soft Jaccard 損失＋BPR 損失で微分可能に同時最適化。
- ユーザーの嗜好空間において崩してよい次元と守るべき次元を自動分類し、独自カバーの GT 命中率を高める。

### 12C: Semantic-aware Soft Jaccard
- アイテム埋め込み間の類似度行列 $S_{ij}$ を挟んだ Soft Jaccard 損失。
- アイテム ID の不一致だけでなく、意味的・ジャンル的な被りを緩和する。

### 12D: 統合評価 & ベースライン比較
- MovieLens 1M (K=10, N_trials=10, Seeds=5)
- Plan 009/010/011 の主要モデル（PostNoise, soft_jaccard, DPP, MultiHead）と同一軸で統一プロット比較。

---

## 期待される成果
- `recall_cum` を `0.0627` 以上に維持しながら、`recall_avg`（1試行あたりの精度）および独自カバーの命中的精度を顕著に改善するモデルの確立。
