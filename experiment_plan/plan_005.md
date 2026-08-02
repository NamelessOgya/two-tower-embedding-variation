# Plan 005: ベースモデル強化 (Whitening, Log-Q Correction, CLIP Logit Scaling) の Ablation Study および M4 多様性検証

実験日: 2026-08-02 | Dataset: MovieLens 1M | Seeds: 5 | K=10 | N_trials=5

---

## 背景・動機

Plan 001〜004 では、未チューニングの言語モデル (mE5) によるテキスト埋め込みのままで推薦を行っていたため、決定論的検索 (M0 Baseline) の精度 (`recall_avg` = 0.0016) が低く、特定アイテムへの偏調（Coverage 5.4%）が顕著であった。

本実験では、基盤モデルの表現力・判別力を向上させる3つの強力な要素技術を導入する:
1. **埋め込みの白色化 (Embedding Whitening / WhitenRec)**: 埋め込み空間の異方性 (Anisotropy) および Hubness を除去する。
2. **Log-Q 補正 (Popularity Bias Correction)**: 頻出人気アイテムによる推薦偏重を抑制する。
3. **CLIP Logit Auto-Scaling ($\frac{1}{\tau}$ 温度スケール)**: コサイン類似度 $[-1, 1]$ と人気度ペナルティ $[-10, -2]$ のスケール不均衡を解消する。

---

## Sub-exp 5A: ベースモデル強化の Ablation Study

以下の 5 つのモデルバリエーションを比較評価し、どの要素技術の組み合わせが最も高い推薦性能を発揮するかを検証する。

| モデル名 | Whitening (白色化) | Log-Q 補正 | Logit Scaling ($\frac{1}{\tau}$) | 説明 |
|---|---|---|---|---|
| `M0_raw` | ✕ | ✕ | ✕ | 従来の未補正 mE5 Baseline |
| `M0_whiten` | ○ | ✕ | ✕ | ZCA 白色化のみ適用 |
| `M0_logq` | ✕ | ○ ($\alpha=0.1$) | ✕ | 単純 Log-Q 補正 (スケール調整なし) |
| `M0_scaled_logq` | ✕ | ○ ($\alpha=0.1$) | ○ ($S=14.3$) | Logit Scaling 付き Log-Q 補正 |
| `M0_strong` | ○ | ○ ($\alpha=0.1$) | ○ ($S=14.3$) | 白色化 + Logit Scaling + Log-Q 補正の統合型 |

期待される観察:
- 白色化により Hubness が除去され、`recall_avg` が大きく向上する。
- Logit Scaling なしの `M0_logq` はスコア不均衡により精度が低迷するが、Logit Scaling を加えた `M0_scaled_logq` および統合型 `M0_strong` で大幅な精度向上が見込まれる。

---

## Sub-exp 5B: 5A ベストモデルに対する M4 Gaussian $\sigma$ スイープ

Sub-exp 5A で最も高精度 (`recall_avg` / `recall_cum`) であった最優秀モデルを自動選択し、そのモデルに対して M4 Gaussian ノイズを付与した場合の精度-多様性トレードオフを検証する。

- ノイズスケール: $\sigma \in \{0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20\}$
- 目的: 「強力なベースモデル」においても、多様性ノイズによって `recall_cum`（N試行累積精度）が拡大するか、および Trade-off 曲線の挙動を確認する。

---

## 実験コード

- モデル定義: `src/model/models_005.py`
- ランナー: `src/run_experiment_005.py`

```bash
# 全実行
python src/run_experiment_005.py --subexp all --device cuda
```
