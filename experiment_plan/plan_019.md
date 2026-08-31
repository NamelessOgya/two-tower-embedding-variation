# Plan 019: テキスト埋め込みモデルのスケール検証 (mE5-small / base / large)

## 1. 目的・背景
Two-Tower モデルにおいて最高精度を達成した **`TwoTowerLogQInfoNCE` ($\tau=0.07, \alpha=1.0$)** を標準アーキテクチャに据え、入力となるテキスト埋め込みモデルのスケール（モデルパラメータ数・表現次元数）を変更した際の推薦性能への影響を検証する。

### リサーチクエスチョン (RQ)
- **RQ1 (精度スケーリング則)**: 埋め込みモデルのサイズ（small ➔ base ➔ large）拡大に伴い、推薦精度（Recall@10, Precision@10, NDCG@10）は単調向上するか？
- **RQ2 (カタログ多様性・バイアス)**: 表現力が高まることで、幾何学的空間の Hubness やカタログ網羅率（Coverage@10, Gini Index）はどのように変化するか？
- **RQ3 (費用対効果)**: 計算リソース（埋め込み生成時間・GPUメモリ）と推薦精度のトレードオフにおける最適解はどこか？

---

## 2. 比較する埋め込みモデル (3 スケール)

1. **`intfloat/multilingual-e5-small`**:
   - 埋め込み次元数: **384 次元**
   - パラメータ数: **約 118M**
2. **`intfloat/multilingual-e5-base`** (現行デフォルト):
   - 埋め込み次元数: **768 次元**
   - パラメータ数: **約 278M**
3. **`intfloat/multilingual-e5-large`**:
   - 埋め込み次元数: **1024 次元**
   - パラメータ数: **約 560M**

---

## 3. 実験プロトコル

- **下流モデル**: `TwoTowerLogQInfoNCE`（$D_{\text{in}} \to 64$ MLP 射影、ZCA Whitening、$\tau=0.07, \alpha=1.0$、25 エポック）
- **データセット**: MovieLens 1M（3,706 映画 / 6,040 ユーザー）
- **評価方法**: 5 つの独立シード（42, 43, 44, 45, 46）の平均値 ± 標準偏差（エラーバー付きプロット）
- **評価指標**:
  - 精度: `Recall@10`, `Precision@10 (%)`, `NDCG@10`, `Hit@10 (%)`
  - 多様性・公平性: `Coverage@10 (%)`, `Gini Index`, `Shannon Entropy`, `Long-tail Coverage (%)`

---

## 4. 成果物
- 埋め込みスクリプト: `src/data/embed_multiscale.py`
- 実験実行スクリプト: `src/run_experiment_019.py`
- 結果データ: `report/plan_019/results_019.csv`
- グラフ: `report/plan_019/tradeoff_plan_019.png`
- 分析レポート: `insight/plan_019_embedding_scale.md`, `report/plan_019/README.md`
