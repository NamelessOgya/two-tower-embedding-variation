# Plan 016: Two-Stage Plackett-Luce Probabilistic Ranking Sampling

## 1. 目的・背景
実運用の推薦システム（YouTube, Netflix, 開発現場等）では、数百万点の全アイテムから直接確率サンプリングを行うことは計算コスト的・精度的に現実的ではない。

実際の運用アーキテクチャに合わせ、本 Plan 016 では **「2段階推薦システム (Two-Stage Recommendation Architecture)」** を導入する。

- **Stage 1 (近傍探索 / 候補生成)**: Two-Tower (FAISS) により、ユーザー `u` に適合する上位 `M` 件（例: M = 200）の候補プール `C_u` を取得する。
- **Stage 2 (Plackett-Luce 確率的サンプリング)**: 候補プール `C_u` のスコアに基づき、Plackett-Luce (PL) モデル（Gumbel-Top-K サンプリング）を用いて Top-K 件（K = 10）のスレートを各試行で独立生成する。

**「過去の推薦履歴（セッション状態）を一切保持しない完全ステートレス」** の制約下で、この2段階構成が精度（Total Slate Precision）と多様性（Cumulative Recall）をどのように両立できるかを定量評価する。

---

## 2. 提案モデル: `TwoTowerTwoStagePlackettLuce`

### アルゴリズムのフロー
1. **Stage 1 (候補プール生成)**:
   - ユーザー `u` のベクトル `q_u` と FAISS インデックスを用いて、全アイテムからスコア上位 `M` 件（デフォルト: M = 200）の候補集合 `C_u` を取得する。
2. **Stage 2 (Plackett-Luce 確率的ランキング)**:
   - 候補プール `C_u` 内の各アイテム `i` について、適合度スコア `s(u, i) = q_u · i`（LogQ 補正含む）を取得する。
   - 温度パラメータ `tau`（tau > 0）で除算した調整後スコア `s(u, i) / tau` を計算。
   - 各候補アイテムに標準 Gumbel ノイズ `G_i ~ Gumbel(0, 1)` を加算。
   - `G_i` は一様乱数 `U_i ~ Uniform(0, 1)` から `G_i = -log(-log(U_i))` で生成。
   - 加算後スコア `s(u, i) / tau + G_i` が大きい順に Top-K 件（K = 10）を選択して試行 `t` の推薦リストを作成する。

### 特徴
- **実運用適合性**: ボトム 95% の低適合度ノイズアイテムが混入するリスクを Stage 1 で遮断し、ユーザーに真に適合する Top-M 候補内でのみ高品質な確率的ばらつきを生み出す。
- **過去履歴依存: なし (完全ステートレス)**: 試行 `t` のサンプリングは独立して行われるため、履歴保持サーバーが不要。

---

## 3. 対象比較モデル (5条件)

1. **`TwoTower_d2_h64`**: ベースライン (決定論的 Top-K, tau -> 0)
2. **`TT_divloss_soft_jaccard_l0p1_s0p05`**: soft_jaccard ベースライン (Plan 011 最高値)
3. **`TT_item_partition_n10`**: Random Item Partition ベースライン (Plan 012)
4. **`TT_semantic_stratified_partition_n10`**: 意味論的層化アイテム分割 (Plan 014/015 最高値)
5. **`TwoTowerTwoStagePlackettLuce`**: 新提案 2段階 PL モデル (M=200, tau = 0.2, 0.5, 1.0, 2.0, 5.0 スイープ)

---

## 4. 評価指標

- **`recall_cum`**: 10試行累積 Recall（ユニーク正例カバー率）
- **`recall_avg`**: 1試行平均 Recall
- **`Total Slate Precision (%)`**: 全100枠（10試行 x Top-10）における正例の割合
- **`Gross Hits`**: 全100枠中での平均正解ヒット個数
- **`Hit Ground-Truth Spread (HGTS)`**: ヒットした正解アイテム間の空間分散度
- **`Hit Genre Coverage (HGC)`**: ヒットした正解の平均映画ジャンル数
- **`Temporal Overlap / Diversity`**: 試行間重なり度および多様度

---

## 5. 成果物
- モデルコード: `src/model/models_016.py`
- 実験スクリプト: `src/run_experiment_016.py`
- 結果データ: `report/plan_016/results_016.csv`, `report/plan_016/tradeoff_plan_016.png`
- 分析レポート: `insight/plan_016_plackett_luce.md`
