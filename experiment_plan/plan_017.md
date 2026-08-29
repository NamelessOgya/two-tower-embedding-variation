# Plan 017: 大規模リアルワールドデータ (Yelp Open Dataset) における 2段階 Plackett-Luce & 多様化体系の検証

## 1. 目的・背景
MovieLens 1M データセットにおいて、**「2段階 Plackett–Luce 確率的サンプリング (Stage 1 FAISS M=200 ➔ Stage 2 Gumbel-Top-K)」** および **「Item Partition (層化分割)」** が、過去履歴非依存（完全ステートレス）の制約下で Total Slate Precision（全スロット精度 +42% UP）と Cumulative Recall（13.2倍 UP）の最高水準を達成した。

本 Plan 017 では、データ規模が MovieLens の数倍〜十数倍大きい**大規模リアルワールドデータセット (Yelp Open Dataset, 解凍後 8.7 GB)** にて同一の評価を実施し、手法の一般的汎化性能および大規模環境でのスケーラビリティを定量検証する。

---

## 2. 実験データセット設定

- **データソース**: Yelp Open Dataset (`data/raw_yelp/`)
- **ドメイン**: レストラン・店舗属性およびユーザーレビュー
- **フィルタリング条件**:
  - レストランカテゴリ限定 (`business.categories` に "Restaurants" を含む)
  - 20-core フィルタリング (User レビュー数 >= 20, Item レビュー数 >= 20)
  - 正解判定基準: 星4以上 (rating >= 4)
  - 時系列分割 (ユーザーごとに 8:1:1 split)
- **テキスト埋め込み**: `intfloat/multilingual-e5-base` (768次元) + ZCA 白色化

---

## 3. 比較対象モデル (5条件)

1. **`TwoTower_d2_h64`**: ベースライン (多様化なし, tau -> 0)
2. **`TT_divloss_soft_jaccard_l0p1_s0p05`**: soft_jaccard ベースライン
3. **`TT_item_partition_n10`**: Random Item Partition ベースライン (n=10)
4. **`TT_semantic_stratified_partition_n10`**: 意味論的層化アイテム分割 (n=10)
5. **`TwoTowerTwoStagePlackettLuce`**: 2段階 PL モデル (M=200, tau = 1.0, 2.0, 5.0 スイープ)

---

## 4. 評価指標 (MovieLens と完全同一の 8 指標)

- **`recall_cum`**: 10試行累積 Recall（ユニーク正例カバー率）
- **`recall_avg`**: 1試行平均 Recall
- **`Total Slate Precision (%)`**: 全100枠における正例のヒット割合
- **`Gross Hits`**: 全100枠中での平均正解ヒット個数
- **`Hit Ground-Truth Spread (HGTS)`**: ヒットした正解店舗間の空間分散度
- **`Hit Category Coverage (HCC)`**: ヒットした正解店舗の平均カテゴリ数
- **`Temporal Overlap / Diversity`**: 試行間重なり度および多様度
- **`ILS (Intra-List Similarity)`**: リスト内類似度

---

## 5. 成果物
- 前処理データ: `data/processed_yelp/`
- 埋め込みデータ: `data/processed_yelp/user_embeddings.npy`, `item_embeddings.npy`
- モデル・実験コード: `src/run_experiment_017.py`
- 評価レポート: `report/plan_017/results_yelp.csv`, `report/plan_017/tradeoff_plan_017_yelp.png`
- インサイト分析: `insight/plan_017_cross_dataset.md`
