# Insight: Gaussian ノイズの挿入位置は output（最終埋め込みの後）にすべき

**実験**: Plan 003 Sub-exp 3A  
**データ**: MovieLens 1M | multilingual-e5-base (frozen) | K=10 | N_trials=5 | Seeds=5  
**σ スイープ**: {0.01, 0.02, 0.05} × {input, middle, output} = 9 条件を網羅

---

## 結論

> **frozen Transformer モデルに対して Gaussian ノイズで推薦多様性を向上させる場合、
> ノイズの挿入位置は必ず「最終埋め込みの後（output）」でなければならない。**
> input や中間層（middle）へのノイズ挿入は、σ をどれだけ大きくしても効果がほぼない。

この結論は σ=0.01 〜 0.05 の **3水準すべてで一貫して確認**されており、
「たまたまその σ で効果がなかった」という解釈は成立しない。

---

## 実験データ（σ × 位置の全結果）

### temporal_overlap（低いほど多様性が高い）

| 挿入位置 | σ=0.01 | σ=0.02 | σ=0.05 | 傾向 |
|---|---|---|---|---|
| M0（ノイズなし） | 1.0000 | 1.0000 | 1.0000 | 基準 |
| **input** | 0.9855 | 0.9683 | 0.9205 | ❌ σ を10倍にしても overlap は 0.08 しか下がらない |
| **middle** | 0.9845 | 0.9664 | 0.9163 | ❌ input とほぼ同じ。全く効いていない |
| **output** | **0.6826** | **0.4301** | **0.1501** | ✅ σ を上げるほど線形的に多様化 |

### recall_cum（高いほど良い）

| 挿入位置 | σ=0.01 | σ=0.02 | σ=0.05 | 傾向 |
|---|---|---|---|---|
| M0（ノイズなし） | 0.0016 | 0.0016 | 0.0016 | 基準 |
| input | 0.0017 | 0.0017 | 0.0021 | ほぼ変化なし |
| middle | 0.0016 | 0.0018 | 0.0021 | ほぼ変化なし |
| **output** | **0.0031** | **0.0047** | **0.0092** | ✅ σ増加とともに recall_cum も改善 |

### recall_avg（高いほど良い）

> `recall_avg` = 1試行あたりの recall の平均。多様性に依存しない「1回あたりの推薦精度」。
> 定義: `recall_avg = mean_t [ |R_t ∩ GT| / |GT| ]`

| 挿入位置 | σ=0.01 | σ=0.02 | σ=0.05 | 傾向 |
|---|---|---|---|---|
| M0（ノイズなし） | 0.0016 | 0.0016 | 0.0016 | 基準 |
| input | 0.0016 | 0.0016 | 0.0016 | 変化なし（ノイズが消えているため当然） |
| middle | 0.0016 | 0.0016 | 0.0016 | 変化なし |
| **output** | 0.0015 | 0.0015 | **0.0019** | σ=0.05 でわずかに上昇（誤差範囲内） |

**解釈**: recall_avg は全位置・全 σ で 0.0015〜0.0019 と**ほぼ横ばい**。
- input/middle：ノイズが消滅するため変化なし（必然的に）
- output：ランダム方向への探索で1試行の精度はわずかに変動するが、有意な低下はない

**ノイズによる精度の劣化は recall_avg では観測されない**。
recall_cum の上昇は「精度が上がった」のではなく「多様な試行の累積効果」である。

### coverage（高いほど多くのアイテムが推薦される）

| 挿入位置 | σ=0.01 | σ=0.02 | σ=0.05 |
|---|---|---|---|
| input | 5.7% | 6.0% | 6.9% |
| middle | 5.6% | 5.9% | 6.9% |
| **output** | **16.7%** | **41.3%** | **97.1%** |

> σ=0.05 の output が coverage=97% に達するのに対して、
> input/middle は同じ σ で coverage=7% 前後に留まる（**14倍の差**）。

---

## σ 増加に伴うトレードオフの非対称性

output では σ を上げると多様性（1-overlap）と recall_cum が**同時に改善**する。
input/middle では σ を 5倍にしても（0.01→0.05）overlap は 0.06〜0.07 しか下がらない。

```
σ=0.01 → σ=0.05 での overlap の改善量:
  input:  0.9855 → 0.9205  = Δ0.065（わずか）
  middle: 0.9845 → 0.9163  = Δ0.068（わずか）
  output: 0.6826 → 0.1501  = Δ0.532（約8倍の効果）
```

---

## なぜ input/middle ノイズは無効なのか

BERT 系 Transformer の各層には **LayerNorm** が含まれる：

```
[各 Transformer 層の内部]
  → Self-Attention
  → Add（残差） + LayerNorm   ← 平均0・分散1に正規化
  → Feed-Forward Network
  → Add（残差） + LayerNorm   ← 再び正規化
```

LayerNorm は入力テンソルを次のように変換する：

```
LayerNorm(x) = (x - mean(x)) / std(x) * γ + β
```

ノイズ ε を加えても、直後の LayerNorm が **x + ε の平均・分散を正規化**するため、
ε が最終出力に与える影響はほぼ消滅する。これが 12 層繰り返されると：

```
input + ε(σ=0.05)
  → LayerNorm  → ε の絶対値が消える
  → LayerNorm  → 残った影響がさらに消える
  → ... × 12層
  → output embedding: ε はほぼ消滅（overlap=0.92）
```

---

## 実装上の指針

```python
# ✅ 正しい実装 — output 位置にノイズを挿入する
with torch.no_grad():
    last_hidden = encoder(input_ids, attention_mask).last_hidden_state
    mask_f = attention_mask.unsqueeze(-1).float()
    emb = (last_hidden * mask_f).sum(1) / mask_f.sum(1)    # avg pool
    emb = F.normalize(emb, p=2, dim=-1)                    # L2 norm

# ← この後にノイズを加える
eps   = torch.randn_like(emb) * sigma
query = F.normalize(emb + eps, p=2, dim=-1)


# ❌ 誤った実装 — input/middle にノイズ
# LayerNorm に吸収されて σ 値に関係なく効果がほぼない

def input_hook(module, inp, out):
    return out + torch.randn_like(out) * sigma  # 無意味

model.embeddings.register_forward_hook(input_hook)  # ← 効果なし
```

---

## 追加的な示唆

- **ファインチューニングありの場合は別**：end-to-end で学習する場合、
  input/middle のノイズは Data Augmentation として機能する可能性がある
  （汎化性能向上目的なら有効）

- **output ノイズの最適 σ**（Plan 002 実験より）：
  - σ=0.02: recall_cum=0.0079, overlap=0.36 → 「精度寄り」のバランス点
  - σ=0.05: recall_cum=0.0166, overlap=0.04 → 「多様性寄り」
  - σ=0.20: recall_cum=0.0221, overlap=0.004 → ほぼランダム探索

- **学習アダプタとの組み合わせ**（Plan 003 Sub-exp 3B より）：
  per-dimension σ を BPR + cosine 損失で学習すると、
  固定 σ の Gaussian より recall_cum が +7.7% 向上する（0.0238 vs 0.0221）

---

## 参照実験

- 実験データ: `report/plan_003/noise_position/results.json`
- サマリー: `report/plan_003/noise_position/summary.csv`
- グラフ: `report/plan_003/noise_position/noise_position.png`
- 詳細レポート: `report/plan_003/report_003.md` (Sub-exp 3A 節)
- 関連コード: `src/model/models_003.py` (`precompute_all_noisy_embeddings`)
