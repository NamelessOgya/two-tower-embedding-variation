# Plan 020: 次世代 LLM 埋め込みモデル比較検証 (Qwen3 / Qwen2 / mE5)

## 1. 目的・背景
従来の BERT/RoBERTa アーキテクチャに基づく埋め込みモデル（`multilingual-e5`）に加え、最新の大規模言語モデル（LLM）基盤の埋め込みモデルである **`Qwen3-Embedding-0.6B`** および **`GTE-Qwen2-1.5B-instruct`** を導入し、Two-Tower 推薦システムにおける世代間・モデルアーキテクチャ間の推薦精度・カタログ多様性を比較検証する。

### リサーチクエスチョン (RQ)
- **RQ1 (世代間アーキテクチャの比較)**: BERT 系（mE5）対 Decoder/LLM 系（Qwen3 / Qwen2）で推薦精度（Recall/Precision/NDCG）に有意な差が生じるか？
- **RQ2 (モデルサイズと次元数のスケーリング)**: 384d (118M) ➔ 768d (278M) ➔ 1024d (560M/600M) ➔ 1536d (1.5B) におけるスケーリング挙動。
- **RQ3 (カタログ多様性とロングテール)**: LLM の高度な文脈理解・知識抽出能力が、推薦のロングテール網羅率（Long-tail Coverage）や Hubness 抑制に寄与するか？

---

## 2. 比較する埋め込みモデル諸元

```
┌─────────────────────────────────┬──────────┬─────────────┬───────────────────────────┐
│ モデル名 (Hugging Face)         │ 埋め込み次元 │ パラメータ数 │ アーキテクチャ / 世代     │
├─────────────────────────────────┼──────────┼─────────────┼───────────────────────────┤
│ intfloat/multilingual-e5-small  │ 384 次元 │ 約 118M     │ Encoder-only (RoBERTa)    │
│ intfloat/multilingual-e5-base   │ 768 次元 │ 約 278M     │ Encoder-only (RoBERTa)    │
│ intfloat/multilingual-e5-large  │ 1024次元 │ 約 560M     │ Encoder-only (RoBERTa)    │
│ Qwen/Qwen3-Embedding-0.6B       │ 1024次元 │ 約 600M     │ Decoder-based (Qwen3 LLM) │
│ Alibaba-NLP/gte-Qwen2-1.5B      │ 1536次元 │ 約 1.5B     │ Decoder-based (Qwen2 LLM) │
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
- 埋め込み生成: `src/data/embed_qwen.py`, `src/data/embed_qwen2_1p5b.py`
- ベンチマーク実行: `src/run_experiment_020.py`
- 結果データ: `report/plan_020/results_020.csv`
- グラフ: `report/plan_020/tradeoff_plan_020.png`
- 報告書: `insight/plan_020_qwen_embeddings.md`, `report/plan_020/README.md`
