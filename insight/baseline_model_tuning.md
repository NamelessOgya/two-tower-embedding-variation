# Insight: ベースモデルの強化手法（Embedding Whitening & CLIP Logit Scaling）

**実験**: Plan 005 Sub-exp 5A & 5B  
**データ**: MovieLens 1M | multilingual-e5-base (frozen) | K=10 | N_trials=5 | Seeds=5  
**対象**: `M0_raw`, `M0_whiten`, `M0_logq`, `M0_scaled_logq`, `M0_strong`, および Gaussian ノイズ $\sigma$ スイープ

---

## 結論

> **事前学習言語モデル（mE5等）のテキスト埋め込みを推薦モデルのベースラインとして使用する場合、
> 「ZCA Whitening（白色化）」と「CLIP スタイルの Logit Scaling 付き Log-Q 補正」の組み合わせにより、
> 未補正モデルに対して精度（recall_avg）を +156%（2.5倍以上）、カバー率（coverage）を 5.5倍に向上させることができる。**

1. **Embedding Whitening の効果**: 埋め込み空間の「異方性（Anisotropy）」と「Hubness（特定アイテムへの偏重）」を除去し、正当な幾何的類似度検索を可能にする。
2. **CLIP Logit Auto-Scaling の不可欠性**: コサイン類似度 $[-1, 1]$ と人気度ペナルティ $[-10, -2]$ のスケール不均衡を、温度スケール $\frac{1}{\tau} \approx 14.3$ によって補正することが必須である。
3. **強モデルにおける Trade-off の顕在化**: ベースモデルが強力（`M0_strong`）になると、ノイズ増加に伴って 1 試行精度（`recall_avg`）が低下する「真の Precision-Diversity Trade-off」が成立する。
4. **Sweet Spot ($\sigma=0.010$)**: 微小ノイズ $\sigma=0.010$ を適用することで、1試行精度を一切損なわずに累積 Recall (`recall_cum`) を **2倍 (+105%)** に向上させることができる。

---

## 実験データ

### 1. Sub-exp 5A: ベースモデル強化 Ablation Study

| モデル名 | Whitening | Log-Q | Logit Scale ($S=\frac{1}{\tau}$) | recall_cum↑ | recall_avg↑ | Hit@10↑ | NDCG@10↑ | Overlap↓ | Coverage↑ |
|---|---|---|---|---|---|---|---|---|---|
| `M0_raw` (従来の未補正) | ✕ | ✕ | — | 0.0016±0.0000 | 0.0016±0.0000 | 1.51% | 0.0020 | 1.0000 | 5.4% |
| `M0_whiten` | ○ | ✕ | — | 0.0028±0.0000 | 0.0028±0.0000 | 2.42% | 0.0034 | 1.0000 | 15.6% |
| `M0_logq` | ✕ | ○ ($\alpha=0.1$) | 1.0 (未調整) | 0.0016±0.0000 | 0.0016±0.0000 | 1.42% | 0.0021 | 1.0000 | 48.5% |
| `M0_scaled_logq` | ✕ | ○ ($\alpha=0.1$) | 14.3 (CLIP風) | 0.0033±0.0000 | 0.0033±0.0000 | 2.70% | 0.0036 | 1.0000 | 17.1% |
| **`M0_strong` (統合型)** | **○** | **○ ($\alpha=0.1$)** | **14.3 (CLIP風)** | **0.0041**±0.0000 | **0.0041**±0.0000 | **3.35%** | **0.0047** | **1.0000** | **29.7%** |

### 2. Sub-exp 5B: 最優秀モデル (`M0_strong`) における M4 Gaussian スイープ

| $\sigma$ | recall_cum↑ (N試行累積) | recall_avg↑ (1試行精度) | Hit@10↑ | NDCG@10↑ | Overlap↓ | Coverage↑ |
|---|---|---|---|---|---|---|
| **0.000** (M0_strong) | 0.0041 | 0.0041 | 3.35% | 0.0047 | 1.0000 | 29.7% |
| **0.001** | 0.0045 | 0.0041 | 3.35% | 0.0047 | 0.9635 | 30.1% |
| **0.005** | 0.0062 | 0.0041 | 3.33% | 0.0046 | 0.8308 | 32.4% |
| **0.010** | **0.0084** | **0.0041** | **3.32%** | **0.0045** | **0.6697** | **38.1%** |
| **0.020** | 0.0121 | 0.0040 | 3.21% | 0.0044 | 0.4009 | 55.2% |
| **0.050** | 0.0197 | 0.0032 | 2.61% | 0.0035 | 0.1060 | 91.7% |
| **0.100** | 0.0219 | 0.0026 | 2.23% | 0.0029 | 0.0185 | 100.0% |
| **0.200** | 0.0223 | 0.0023 | 2.05% | 0.0026 | 0.0048 | 100.0% |

---

## メカニズム解説

### 1. ZCA Whitening（白色化）が効く理由

事前学習言語モデルの出力ベクトル空間は、すべてのベクトルが狭い角度空間内に密集する「異方性（Anisotropy）」の性質を持っています。
これにより、本来無関係なアイテム同士の類似度が高くなり、幾何学的な中心点近くに位置する「Hub アイテム」ばかりが過剰にTop-$K$に入り込みます。

ZCA Whitening 変換 $e' = \text{L2\_norm}((e - \mu) W)$（ただし $W = U \Lambda^{-1/2} U^T$）を適用することで：
- 埋め込みの平均 $\mu$ が除外される
- 各軸の分散が均一化（等方化 / Isotropic）される
- Hub アイテムの無駄な優位性が排除され、アイテム本来の意味的類似度が正しく評価されるようになります。

### 2. CLIP Logit Auto-Scaling が不可欠な理由

人気度バイアス補正（Log-Q Correction）では、アイテムの出現確率 $q_i$ の対数ペナルティを引いてスコア化します：
$$\text{score}(u, i) = \text{scale} \cdot \langle e_u, e_i \rangle - \alpha \log(q_i)$$

コサイン類似度 $\langle e_u, e_i \rangle$ の生の値域は $[-1, 1]$（実際の上位候補では $0.2 \sim 0.6$ 程度）と極めて狭い一方、$\log(q_i)$ は $[-9.2, -4.2]$ と広い幅を持ちます。

- スケール未調整（$\text{scale}=1.0$）の場合：人気度ペナルティが類似度を圧倒してしまい、セマンティクス（ユーザー嗜好）が無視されます（`M0_logq` の recall が改善しない原因）。
- CLIP 風の Logit Scaling（$\text{scale} = \frac{1}{\tau} = 14.3$, $\tau=0.07$）の適用時：コサイン類似度スコアが $[-14.3, 14.3]$ に拡張され、セマンティックな適合度と人気度ペナルティが初めて対等かつ最適に融合します。

---

## 推奨実装パターン

事前学習テキスト埋め込みベースの推薦モデルにおける推奨構成：

```python
# 1. ZCA Whitening
whitener = Whitener()
whitener.fit(item_embeddings)
item_embs_white = whitener.transform(item_embeddings)
user_embs_white = whitener.transform(user_embeddings)

# 2. Log-Q Popularity Corrector + CLIP Scaling
logq_corrector = LogQCorrector(train_interactions, n_items, alpha=0.1)
logit_scale = 14.3  # tau = 0.07

# 3. 推論時スコア計算
cand_scores = (logit_scale * (user_emb @ item_embs_white.T)) - (0.1 * logq_corrector.log_q)

# 4. 多様化用ノイズ（必要に応じて）
# user_emb_noisy = normalize(user_emb + N(0, 0.01^2))
```
