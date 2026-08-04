# Plan 013: Hypothesis Verification — Mechanism of Item Partition Superiority

## 目的・背景
Plan 012 において、ユーザー提案の新ベースライン **Item Partition (`n=10`)** が、学習系多様化モデル (soft_jaccard) や推論ノイズ系モデル (PostNoise) を圧倒し、`recall_cum = 0.1172` (元の1.87倍)、`recall_avg = 0.0117` (ベースライン以上) という驚異的な最高性能を叩き出した。

本 Plan 013 では、以下の仮説：
> **「全アイテム検索（ベースライン）は同質アイテムで Top-10 の枠が占領（冗長化）されているのに対し、Item Partition は枠の占領を解除し、ユーザーの持つ複数の異なる Ground Truth（正解群）を効率よく救い出している」**

を、3 つの定量的評価指標と可視化分析によって実証・証明する。

---

## 定量的検証指標

### 1. リスト内同質度 (Intra-List Similarity: ILS)
- 推薦された Top-10 リスト内におけるアイテムペア間の平均コサイン類似度（mE5 768次元埋め込み）。
- **仮説**: ベースラインは `ILS` が高く（特定ジャンルで枠が占領）、Item Partition は `ILS` が低い（同質化が解除）。

### 2. ヒット正解アイテムの空間分散度 (Hit Ground-Truth Spread: HGTS)
- 各ユーザーにおいて 10 試行合計でヒットした Ground Truth アイテム同士の非類似度（$1 - \text{cosine\_sim}$）の平均。
- **仮説**: ベースラインは `HGTS` が低く（特定の正解クラスタに集中）、Item Partition は `HGTS` が高い（多様な正解クラスタに跨がってヒット）。

### 3. ヒット正解アイテムのジャンルカバー数 (Hit Genre Coverage: HGC)
- 10 試行合計でヒットした Ground Truth アイテムが含むユニーク映画ジャンル数（例: Action, Drama, Comedy 等）。
- **仮説**: Item Partition の方がより幅広いジャンルの正解をヒットさせている。

---

## 対象比較モデル (4モデル)
1. **TwoTower (no-div)**: ベースライン
2. **PostNoise (σ=0.20)**: 推論ノイズ型
3. **soft_jaccard (λ=0.1, σ=0.05)**: 学習損失型
4. **Item Partition (n=10)**: アイテム 10 分割型

---

## 成果物
- 評価数値サマリー (`report/plan_013/results.json`, `report/plan_013/hypothesis_metrics.csv`)
- 仮説検証プロット (`report/plan_013/hypothesis_verification_013.png`)
- 検証結果レポート (`insight/hypothesis_partition_mechanism.md`)
