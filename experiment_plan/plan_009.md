# Plan 009: soft_jaccard λ 精密スイープ（Two-Tower + DivLoss 最適化）

実験日: TBD | Dataset: MovieLens 1M | Seeds: 5 | K=10 | N_trials=10

---

## 背景・動機

Plan 008 により、`TT_divloss_soft_jaccard` λ=0.1 が recall_cum=0.0615 で全条件中の最高値を記録した。

しかし λ=0.1 の時点でも **λ の減少に伴う recall_cum の単調増加トレンドが続いている**。

| λ | recall_cum | hit@10 | diversity |
|---|---|---|---|
| 0（base TT） | 0.0111 | 0.0892 | 0.000 |
| **0.1** | **0.0615** | 0.0738 | 0.902 |
| 0.5 | 0.0559 | 0.0583 | 0.950 |
| 1.0 | 0.0509 | 0.0529 | 0.957 |

重要な観察：

- λ=0.1 でも **diversity=0.90 と非常に高く**、これは推論時ノイズ（σ=0.05）が主役になっている可能性が高い
- つまり λ を極小にしても推論ノイズが多様性を担保するため、λ を下げるほど hit@10 を回復できる可能性がある
- λ=0 に向かうにつれ base TT（recall_cum=0.0111, diversity=0）に近づくが、
  その手前に「高 recall_cum × 高 diversity」の最適点が存在すると推測される

---

## 実験構造

### Sub-exp 9A: soft_jaccard λ 精密スイープ

- λ ∈ {0.001, 0.005, 0.01, 0.03, 0.05, 0.1}（plan_008 の λ=0.1 以下を細かく探索）
- σ = 0.05（plan_008 最良値で固定）
- base: TT_d2_h64（plan_008 と同様に再学習）
- epochs=30, lr=2e-3, batch_size=512

### Sub-exp 9B: kl_dist バグ修正版の再実験

Plan 008 で `kl_dist` が一貫して他手法より低精度だった原因を調査した結果、
[`models_003.py` L226](file:///home/kasumi/tt-embedding-variation/src/model/models_003.py#L226) に
**二重符号反転バグ** が発見された。

```python
# バグ（現状）
kl = -(log_p1 * p2).sum(0).mean()   # = -KL
return -kl                            # = -(-KL) = +KL  ← KLを最小化（多様性↓）

# 正しい実装
kl = -(log_p1 * p2).sum(0).mean()   # = -KL
return kl                             # = -KL  ← minimize で KL↑（多様性↑）
```

修正後の kl_dist を Plan 008 と同一条件で再実験する：
- λ ∈ {0.1, 0.5, 1.0}（plan_008 との直接比較）
- σ = 0.05

---

## 全体傾向の考察（Plan 008 グラフより）

### diversity vs recall_cum（左グラフ）
- **soft_jaccard を除く全手法で右肩上がり**の傾向 →
  diversity が上がるほど N-trial での累積カバレッジが増加する（直感に一致）
- soft_jaccard は λ=0.1 でもグラフ右上（高 diversity × 高 recall_cum）に位置しており、
  λ をさらに小さくすれば右上を維持しつつ recall_cum がさらに伸びる可能性がある

### diversity vs recall_avg（右グラフ）
- **全手法で右下がりの曲線** → 多様化を強めるほど 1-trial あたりの精度は低下するトレードオフが存在する
- これは本質的な制約：1試行で当てる確率と、N試行で網羅する確率は相反する

### kl_dist の異常
- 他手法と比べて精度が一貫して低く、かつ **λ が増えるほど diversity が下がる**という
  逆方向の挙動（λ=0.1: ov=0.21、λ=0.5: ov=0.38）→ バグによる多様性の逆方向学習が原因

---

## 仮説

λ ∈ {0.01, 0.03} 付近に recall_cum のピークが存在し、
**recall_cum > 0.065 かつ hit@10 > 0.08** が狙える可能性がある。

---

## 実験コード

- モデル定義: `src/model/models_008.py`（TwoTowerDivLoss を流用）
- ランナー: `src/run_experiment_009.py`
- 出力: `report/plan_009/`

```bash
PYTHONPATH=. python3 src/run_experiment_009.py --subexp all --device cuda
```
