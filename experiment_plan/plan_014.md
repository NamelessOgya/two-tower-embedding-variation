# Plan 014: Stateless High-Precision & High-Diversity Methods (Stratified & Multi-Head Partition)

## 目的・背景
Plan 012/013 において、アイテム空間分割ベースライン **Item Partition (`n=10`)** が `recall_cum = 0.1328` (従来の1.87倍)、`recall_avg = 0.0133` (ベースライン以上)、Total Slate Precision = 1.21% というダントツの最高性能を示した。

本 Plan 014 では、**「過去の推薦履歴（セッション状態）を一切保持・参照しない（完全ステートレス）」** という実運用制約の下で、Item Partition のランダム分割の弱点（特定バケットでのヒット不発）を解決し、Item Partition を超える精度と多様性を達成する 2 つの新手法を提案・実装・比較する。

---

## 提案モデル

### 1. `TT_semantic_stratified_partition_n10` (手法 1: 意味論的・層化アイテム分割)
- **概要**: アイテム埋め込み空間上で K-Means クラスタリング ($K_{cls}=20$) を行い、各クラスタから均等（Stratified）に $N=10$ 個のバケットへ分配。
- **狙い**: すべてのバケットに全映画ジャンル/全クラスタの代表品が均等に含まれるため、毎回の枠精度（Total Slate Precision）が確実に向上する。

### 2. `TT_multihead_stratified_partition_n10` (手法 2: マルチヘッド直交クエリ × 意味論的層化分割)
- **概要**: 意味論的層化バケット分割に、ユーザーエンコーダー側のマルチヘッド直交射影 ($W_1, \dots, W_{10}$) を組み合わせる。
- **狙い**: 第 $t$ 試行では第 $t$ ヘッドで生成された多角的なユーザー利得ベクトル $\mathbf{q}_t = W_t \mathbf{u}$ で第 $t$ バケット内を Top-K 検索。重複ゼロを保ちつつ、各バケットでの適合精度を高める。

---

## 対象比較モデル (5モデル)
1. `TwoTower_d2_h64`: ベースライン (no-div)
2. `TT_divloss_soft_jaccard_l0p1_s0p05`: soft_jaccard ベースライン
3. `TT_item_partition_n10`: Random Item Partition ベースライン
4. `TT_semantic_stratified_partition_n10`: Plan 014 手法 1
5. `TT_multihead_stratified_partition_n10`: Plan 014 手法 2

---

## 成果物
- 実装コード: `src/model/models_014.py`, `src/run_experiment_014.py`
- 評価レポート: `report/plan_014/results_014.csv`, `report/plan_014/tradeoff_plan_014.png`
- インサイト文書: `insight/plan_014_stateless_methods.md`
