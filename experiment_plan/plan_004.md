# Plan 004: 多様性損失アダプタの λ スイープ / 全手法統合トレードオフ比較

実験日: 2026-07-31 | Dataset: MovieLens 1M | Seeds: 5 | K=10 | N_trials=5

---

## 背景・動機

Plan 003 Sub-exp 3B では多様性損失の種類を比較したが、λ（損失の重み係数）は
全手法で 1.0 に固定していた。λ の大きさで精度-多様性トレードオフを制御できるはず
なので、λ スイープを実施してフロンティア曲線を描く。

また、これまで計画ごとにバラバラだったプロット（σ スイープ, 位置比較, 損失比較）を
**1枚の統合トレードオフ図**にまとめ、手法間の相対的な位置を明確にする。

---

## Sub-exp 4A: 多様性アダプタの λ スイープ

```
L = BPR(q1) + BPR(q2) + λ · div_loss(q1, q2)
```

λ = {0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}

- `λ=0.0` = 純粋 BPR のみ（多様性損失なし）のコントロール条件
- `λ=10.0` = 多様性優先の極端な設定

スイープ対象の損失関数（全6種）:

| 損失名 | レベル |
|---|---|
| `cosine_emb` | 埋め込み |
| `l2_emb` | 埋め込み |
| `kl_dist` | スコア分布 |
| `js_dist` | スコア分布 |
| `soft_jaccard` | リスト |
| `listnet` | ソフトランク |

合計: 6 losses × 8 λ = **48 モデル**

期待される観察:
- λ が小さい → recall_avg 高め・overlap 高め（精度寄り）
- λ が大きい → recall_avg 低め・overlap 低め（多様性寄り）
- 各損失関数のフロンティア曲線の形が異なる

詳細: `report/plan_004/lambda_sweep/`

---

## Sub-exp 4B: 全手法統合トレードオフ図

以下を1つのグラフに統合する:

| 系列 | データ元 | 制御パラメータ |
|---|---|---|
| M0 baseline（多様性なし） | Plan 001 | — |
| M4 Gaussian σ スイープ | Plan 002 | σ={0.001〜0.2} |
| 3B cosine_emb λ スイープ | Plan 004 (4A) | λ={0〜10} |
| 3B l2_emb λ スイープ | Plan 004 (4A) | λ={0〜10} |
| 3B kl_dist λ スイープ | Plan 004 (4A) | λ={0〜10} |
| 3B js_dist λ スイープ | Plan 004 (4A) | λ={0〜10} |
| 3B soft_jaccard λ スイープ | Plan 004 (4A) | λ={0〜10} |
| 3B listnet λ スイープ | Plan 004 (4A) | λ={0〜10} |

グラフは2枚:
1. **recall_avg vs diversity**（1試行あたり精度 × 多様性）
2. **recall_cum vs diversity**（N試行累積 recall × 多様性）

X軸: `1 - temporal_overlap`（値が大きいほど多様）

詳細: `report/plan_004/tradeoff_unified.png`

---

## 実験コード

- ランナー: `src/run_experiment_004.py`
- アダプタモデル: `src/model/models_003.py` (`M_DiversityAdapter`)

```bash
# 4A: λ スイープ実験
python src/run_experiment_004.py --subexp 4a --device cuda

# 4B: 統合プロットのみ（既存データから）
python src/run_experiment_004.py --subexp plot_only

# 全実行
python src/run_experiment_004.py --subexp all --device cuda --n_trials 5
```
