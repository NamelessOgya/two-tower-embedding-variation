# Plan 015: End-to-End Trained Multi-Head Stratified Partitioning

## 目的・背景
Plan 014 で検証した「無学習の幾何学的直交回転マルチヘッド」は、二タワー適合度空間のアライメントを破壊し、精度が `0.0196` へ暴落した。

本 Plan 015 では、**「過去履歴依存ゼロ（完全ステートレス）」** の制約を保持したまま、ユーザーエンコーダーの出力層に構築した $N=10$ 個のマルチヘッド $W_1, \dots, W_{10}$ が、それぞれの担当アイテムバケット $I_1, \dots, I_{10}$ に最適にアラインするように **端対端（End-to-End）ファインチューニング** を行う。

---

## 提案モデル

### `TT_trained_multihead_stratified_partition_n10` (端対端学習型マルチヘッド層化分割)
- **概要**:
  - 全アイテムを K-Means ($K_{cls}=20$) + 層化抽出により $N=10$ 個のバケット $I_1, \dots, I_{10}$ に分配。
  - ユーザーエンコーダーに 10 個の学習可能線形プロジェクション `self.heads = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(10)])` を配置。
  - 学習時、アイテム $i_{\text{pos}} \in I_t$ に対して第 $t$ ヘッド $\mathbf{q}_t = W_t \cdot \text{user\_head}(u)$ を算出し、バケット $I_t$ 内の負例に対する BPR 損失で端対端学習。
- **推論時 (完全ステートレス)**:
  - 試行 $t$ では第 $t$ ヘッド $\mathbf{q}_t = W_t \cdot \text{user\_head}(u)$ で第 $t$ バケット $I_t$ 内を Top-K 検索。過去の推薦履歴は一切使用しない。

---

## 対象比較モデル (4モデル)
1. `TwoTower_d2_h64`: ベースライン (no-div)
2. `TT_item_partition_n10`: Random Item Partition ベースライン
3. `TT_semantic_stratified_partition_n10`: Semantic Stratified Partition (Plan 014 最高値)
4. `TT_trained_multihead_stratified_partition_n10`: Plan 015 端対端学習型マルチヘッド層化分割

---

## 成果物
- 実装コード: `src/model/models_015.py`, `src/run_experiment_015.py`
- 評価レポート: `report/plan_015/results_015.csv`, `report/plan_015/tradeoff_plan_015.png`
- インサイト文書: `insight/plan_015_trained_multihead.md`
