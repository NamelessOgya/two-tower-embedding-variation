# Plan 006: Strong Baseline 上での全多様性手法スイープ（極小 λ スイープ含む）および Pareto Frontier 統合評価

実験日: 2026-08-02 | Dataset: MovieLens 1M | Seeds: 5 | K=10 | N_trials=5

---

## 背景・動機

Plan 005 により、**ZCA Whitening + CLIP Logit Scaling + Log-Q 人気度補正** を組み合わせた強力なベースモデル **`M0_strong`** が構築された。

本実験（Plan 006）では、この **`M0_strong` を基盤モデル** として用い、すべての多様性制御手法（Gaussian ノイズ、MC Dropout、学習型 3B アダプタ）のパラメータスイープを再実施する。
特に 3B 多様性アダプタにおいては、従来の $\lambda \ge 0.1$ での多様度飽和現象を解明・制御するため、**極小領域 ($\lambda \in \{0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0\}$)** での拡張スイープを実施する。

---

## 実験構造

### Sub-exp 6A: M4 Gaussian ノイズ $\sigma$ スイープ on `M0_strong`
- $\sigma \in \{0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20\}$

### Sub-exp 6B: M5 MC Dropout $p$ スイープ on `M0_strong`
- $p \in \{0.05, 0.10, 0.20, 0.30, 0.50\}$

### Sub-exp 6C: 3B Diversity Adapter 極小 $\lambda$ スイープ on `M0_strong`
- 白色化されたユーザー埋め込み上で `M_DiversityAdapter` を訓練
- 損失関数: `cosine_emb`, `l2_emb`, `soft_jaccard`, `listnet`
- 極小 $\lambda$ スイープ範囲: $\lambda \in \{0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0\}$ (9段階)

### Sub-exp 6D: Pareto Frontier 統合評価プロット
- 全手法の `recall_avg` vs `1 - temporal_overlap` および `recall_cum` vs `1 - temporal_overlap` を 1 枚の統合グラフ（`tradeoff_006_unified.png`）に描画。

---

## 実験コード

- モデル定義: `src/model/models_006.py`
- ランナー: `src/run_experiment_006.py`

```bash
PYTHONPATH=. python3 src/run_experiment_006.py --subexp all --device cuda
```
