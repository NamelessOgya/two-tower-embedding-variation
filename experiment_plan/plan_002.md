# Plan 002: 行動履歴不要手法の深掘り実験
## M4 Gaussian σ スイープ / M6 VAE 改良 / M5 Dropout 改良

実験日: 2026-07-30 | Dataset: MovieLens 1M | Seeds: 5 | K=10 | N_trials=10

---

## Sub-exp 2A: M4 Gaussian σ スイープ

σ ∈ {0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}
→ 精度-多様性トレードオフ曲線を可視化

## Sub-exp 2B: M6 VAE 改良

失敗原因: MSE損失（超球面に不適合）+ Posterior collapse（β=1が弱い）

M6a: beta=4 (beta-VAE)
M6b: beta=8
M6c: Cosine損失 beta=1
M6d: Cosine損失 beta=4 (最有力)

## Sub-exp 2C: M5 Dropout 改良

M5a: Rate sweep p∈{0.05,0.1,0.2,0.3,0.5}
M5b: Structured dropout (64次元グループ単位)
M5c: Soft dropout (mask→noise置換)
