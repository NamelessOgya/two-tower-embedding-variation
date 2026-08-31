# Plan 021: Qwen3-Embedding スケーリング検証 (0.6B vs. 4B vs. 8B)

## 1. 目的・背景
LLM 由来の埋め込みモデル **`Qwen3-Embedding`** シリーズにおいて、モデルパラメータ規模（**0.6B / 4B / 8B**）と潜在表現次元（**1024d / 2560d / 4096d**）をスケールアップした際の、Two-Tower 推薦精度（Recall/Precision/NDCG）およびカタログ多様性（Catalog Coverage/Gini）への影響を系統的に検証する。

### リサーチクエスチョン (RQ)
- **RQ1 (LLM 埋め込みのスケーリング則)**: パラメータ規模が 0.6B ➔ 4B ➔ 8B と拡大するにつれて、推薦精度は単調向上するか？
- **RQ2 (超高次元表現の幾何学的健全性)**: 4096 次元などの超高次元埋め込みを ZCA Whitening + MLP 射影した際、Hubness やロングテール網羅率は改善するか？
- **RQ3 (計算リソースと精度の費用対効果)**: 8B などの巨大エンコーダを採用する実務的メリットとトレードオフ。

---

## 2. 比較する Qwen3-Embedding モデル諸元

```
┌──────────────────────────────┬──────────┬─────────────┬───────────────────────────┐
│ モデル名 (Hugging Face)      │ 埋め込み次元 │ パラメータ数 │ 特徴                      │
├──────────────────────────────┼──────────┼─────────────┼───────────────────────────┤
│ Qwen/Qwen3-Embedding-0.6B    │ 1024次元 │ 約 0.6B     │ 軽量・高速・SOTA          │
│ Qwen/Qwen3-Embedding-4B      │ 2560次元 │ 約 4.0B     │ 中規模・高表現力          │
│ Qwen/Qwen3-Embedding-8B      │ 4096次元 │ 約 8.0B     │ 最大規模・最上位モデル    │
└──────────────────────────────┴──────────┴─────────────┴───────────────────────────┘
```

---

## 3. 実験プロトコル

- **下流モデル**: `TwoTowerLogQInfoNCE`（ZCA Whitening、$\tau=0.07, \alpha=1.0$、25 エポック、5 Seeds: 42, 43, 44, 45, 46）
- **データセット**: MovieLens 1M（3,706 映画 / 6,040 ユーザー）
- **評価指標**: 単一試行 Top-10 評価
  - 精度: `Recall@10`, `Precision@10 (%)`, `NDCG@10`, `Hit@10 (%)`
  - 多様性・公平性: `Coverage@10 (%)`, `Gini Index`, `Shannon Entropy`, `Long-tail Coverage (%)`

---

## 4. 成果物
- 埋め込み生成: `src/data/embed_qwen_scales.py`
- ベンチマーク実行: `src/run_experiment_021.py`
- 結果データ: `report/plan_021/results_021.csv`
- グラフ: `report/plan_021/tradeoff_plan_021.png`
- 報告書: `insight/plan_021_qwen_scaling.md`, `report/plan_021/README.md`
