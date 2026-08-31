# Plan 022: F2LLM-v2 埋め込みスケーリング検証 (0.6B vs. 4B vs. 8B)

## 1. 目的・背景
CodeFuse AI が開発した最新オープンソース埋め込みモデル **`F2LLM-v2`**（Foundation to Feature LLM）シリーズを導入し、Two-Tower 推薦システム（Log-Q InfoNCE 学習基盤）における推薦適合精度および多様性を検証する。

また、**`F2LLM-v2`（単一ステージ精緻化）** vs **`Qwen3-Embedding`（公式指示チューニング）** vs **`multilingual-e5`（双方向RoBERTa）** の 3 大ファミリーを同一条件下で横断比較する。

### リサーチクエスチョン (RQ)
- **RQ1 (F2LLM vs Qwen3 vs mE5)**: CodeFuse による単一ステージ精緻化モデル（F2LLM）は、Qwen3 公式埋め込みや mE5 と比較して推薦精度でどのような差異を示すか？
- **RQ2 (F2LLM のスケーリング則)**: 0.6B (1024d) ➔ 4B (2560d) ➔ 8B (4096d) における精度と多様性の変化。
- **RQ3 (三大ファミリーのパレート境界)**: 精度・多様性・計算コストの総合パレート境界における各モデルの位置付け。

---

## 2. 比較するモデル諸元

```
┌─────────────────────────────────┬──────────┬─────────────┬───────────────────────────┐
│ モデル名 (Hugging Face)         │ 埋め込み次元 │ パラメータ数 │ 特徴 / ファミリー         │
├─────────────────────────────────┼──────────┼─────────────┼───────────────────────────┤
│ codefuse-ai/F2LLM-v2-0.6B       │ 1024次元 │ 約 0.6B     │ F2LLM 系列・軽量          │
│ codefuse-ai/F2LLM-v2-4B         │ 2560次元 │ 約 4.0B     │ F2LLM 系列・中規模        │
│ codefuse-ai/F2LLM-v2-8B         │ 4096次元 │ 約 8.0B     │ F2LLM 系列・大規模        │
│ Qwen/Qwen3-Embedding-0.6B       │ 1024次元 │ 約 0.6B     │ Qwen3 系列・軽量          │
│ Qwen/Qwen3-Embedding-4B         │ 2560次元 │ 約 4.0B     │ Qwen3 系列・中規模        │
│ Qwen/Qwen3-Embedding-8B         │ 4096次元 │ 約 8.0B     │ Qwen3 系列・大規模        │
│ intfloat/multilingual-e5-large  │ 1024次元 │ 約 560M     │ mE5 系列・RoBERTa         │
└─────────────────────────────────┴──────────┴─────────────┴───────────────────────────┘
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
- 埋め込み生成: `src/data/embed_f2llm_scales.py`
- ベンチマーク実行: `src/run_experiment_022.py`
- 結果データ: `report/plan_022/results_022.csv`
- グラフ: `report/plan_022/tradeoff_plan_022.png`
- 報告書: `insight/plan_022_f2llm_benchmark.md`, `report/plan_022/README.md`
