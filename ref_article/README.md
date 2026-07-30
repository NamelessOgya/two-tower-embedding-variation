# 先行研究一覧：Two-Tower Modelにおける推薦多様性

## 実験テーマ

**言語情報の埋め込みを使ったTwo-Tower Modelにおいて、被検索側（item）のベクトルを変えることなく、精度を落とさずに推薦多様性を実装する**

すなわち、クエリ側（ユーザー）の表現のみに変化を加え、毎回異なる候補が推薦されるような仕組みを構築すること。

---

## 論文一覧

### カテゴリ1：クエリ側の多ベクトル表現（Multi-Interest / Multi-Embedding）

| タイトル | 会議/年 | ファイル |
|---|---|---|
| MIND: Multi-Interest Network with Dynamic Routing | CIKM 2019 | [mind.md](./mind.md) |
| ComiRec: Controllable Multi-Interest Framework | KDD 2020 | [comirec.md](./comirec.md) |
| PinnerSage: Multi-Modal User Embedding at Pinterest | KDD 2020 | [pinnersage.md](./pinnersage.md) |
| AMER: Beyond Single Embeddings, Multi-Query Retrieval | arxiv 2024 | [amer.md](./amer.md) |
| Pinterest Multi-Embedding (Implicit+Explicit) | KDD 2025 | [pinterest_multi_embedding.md](./pinterest_multi_embedding.md) |

### カテゴリ2：ポスト検索多様化（Post-Retrieval Diversification）

| タイトル | 会議/年 | ファイル |
|---|---|---|
| MMR: Maximal Marginal Relevance | SIGIR 1998 | [mmr.md](./mmr.md) |
| Fast Greedy MAP Inference for DPP | NeurIPS 2018 | [dpp_recommendation.md](./dpp_recommendation.md) |

---

## 実験との関連性まとめ

本実験の制約（**item側ベクトル固定**）を満たす手法は以下の通り：

### ✅ 直接的に適用可能

1. **クエリ側多ベクトル（MIND, ComiRec, PinnerSage 系）**
   - ユーザーを複数のベクトルで表現し、毎回異なるベクトルでANN検索を行う
   - item側は一切変更不要

2. **MMR / DPP（ポスト検索多様化）**
   - ANNで取得した候補リストを多様化して再ランキング
   - item側は一切変更不要

### ⚠️ 参考になるが直接適用が困難

- AMER: クエリ側の変化のみだが、複数クエリの学習に工夫が必要

---

## 調査日

2026-07-30
