# Plan 018: Two-Tower 学習時における人気アイテム偏重・Hubness 抑制手法の比較研究

## 1. 目的・背景
これまでの推論時サンプリング（Plackett-Luce）やカタログ分割では、固定された学習済みモデルの出力を多様化することでスロットの有効活用を達成した。

本 Plan 018 では、**「Two-Tower の学習段階（Training-time）そのものにおいて人気アイテムへの過剰集中（Popularity Bias / Hubness / 幾何学的空間崩壊）を直接抑制する」** 手法を研究する。

累積複数試行ではなく、**「単一の推薦試行（Top-10）における推薦精度（Recall/NDCG）とカタログ全体の網羅性（Catalog Coverage / Gini Index / Shannon Entropy）」** を評価軸とし、学習段階での De-biasing が埋め込み空間の健全性に与える影響を定量化する。

---

## 2. 比較検証する 4 つの学習時 De-biasing アプローチ

1. **`TT_BPR_base` (標準ベースライン)**:
   - 一様ランダム負例サンプリング + ペアワイズ BPR 損失。
2. **`TT_LogQ_InfoNCE` (損失関数内 Log-Q 補正 InfoNCE)**:
   - *Google (RecSys 2019, Yi et al.)*: 損失の分子・分母でアイテム頻度の対数 $\log q_i$ を差し引いて逆伝播。
3. **`TT_Pop_Negative` (人気度偏重負例サンプリング / Mixed Negative Sampling)**:
   - *Google (WWW 2020, Yang et al.)*: 負例アイテム $j$ を一様ではなく出現確率 $P_{\text{neg}}(j) \propto q_j^\beta$（$\beta = 0.5 \sim 0.75$）でサンプリング。
4. **`TT_Uniformity_Loss` (超球面 Uniformity 正則化 Two-Tower)**:
   - *ICML 2020 (Wang & Isola) / DirectAU (KDD 2022)*: アイテム埋め込み間に超球面全体へ広がる斥力損失 $\mathcal{L}_{\text{uniform}} = \log \mathbb{E}[\exp(-2\|\boldsymbol{x}_i - \boldsymbol{x}_j\|^2)]$ を追加。
5. **`TT_Adaptive_Tau` (人気度適応型動的温度 Softmax)**:
   - 人気アイテムほど温度 $\tau_i$ を高く、テールアイテムほど温度を冷やして学習。

---

## 3. 評価指標（単一試行 Top-10 評価）

### 精度指標 (Accuracy)
- **`Recall@10`**: 正例適合アイテムの Top-10 カバー率（全ユーザー平均）
- **`Precision@10`**: 10 枠中の正例適合率
- **`NDCG@10`**: ランキング順位考慮スコア
- **`Hit@10`**: 1 件以上正解が含まれたユーザー割合

### システム全体の多様性・バイアス抑制指標 (Aggregate Diversity & Fairness)
- **`Catalog Coverage@10` [%]**: 全ユーザーの Top-10 に 1 回以上現れたユニークアイテム数 / 全アイテム数
- **`Gini Index` (不均衡度)**: アイテム推薦回数分布のジニ係数（0: 完全均等 ➔ 1: 極度集中）
- **`Shannon Entropy`**: 推薦分布の情報エントロピー $-\sum p_i \log_2 p_i$（大きいほど多様）
- **`Long-tail Coverage` [%]**: 下位 80% のテールアイテムのうち、1 回以上推薦された割合

---

## 4. 対象データセット
1. **MovieLens 1M** (3,706 映画 / 6,040 ユーザー)
2. **Yelp 10-Core** (32,496 店舗 / 73,144 ユーザー / 大規模カタログ・激しい人気バイアス)

---

## 5. 成果物
- モデル定義: `src/model/models_018.py`
- 実行スクリプト: `src/run_experiment_018.py`
- 結果データ: `report/plan_018/results_018_movielens.csv`, `report/plan_018/results_018_yelp.csv`
- グラフ: `report/plan_018/tradeoff_plan_018.png`
- 分析レポート: `insight/plan_018_training_debiasing.md`
