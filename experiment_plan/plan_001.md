# Plan 001: 属性テキストベース Two-Tower Model の精度・多様性ベースライン実験

## 概要

| 項目 | 内容 |
|---|---|
| 実験ID | plan_001 |
| 目的 | 属性情報をテキスト化した Two-Tower Model のベースライン精度・多様性の確認 |
| 埋め込みモデル | multilingual-e5-base |
| データセット | MovieLens 1M / Yelp Open Dataset |
| テキスト言語 | 英語 |
| 正解Ratingしきい値 | ≥ 4（MovieLens）/ ≥ 4（Yelp）|
| 作成日 | 2026-07-30 |
| 最終更新 | 2026-07-30 |

---

## 1. 実験の目的

言語情報埋め込みを用いた Two-Tower Model において、以下を定量的に確認する：

1. **精度確認**：属性テキストを埋め込みにすることで、ユーザー-アイテムの関連性がどの程度捉えられるか（Recall@K）
2. **多様性確認**：同じユーザーに対して複数回推薦したとき、推薦リストがどの程度重複するか（推薦間重複率）
3. **ベースラインの確立**：後続の多ベクトル化・多様性実装実験（plan_002以降）との比較基準を作る

---

## 2. データセット

2つのデータセットを用いて結果の汎化性を確認する。
選定基準：**ユーザー側の属性情報が明示的に取得できること**（Amazon Reviews 2023は購入ログが主でユーザー属性が欠如するため除外）。

---

### データセット 1：MovieLens 1M

| 項目 | 内容 |
|---|---|
| URL | https://grouplens.org/datasets/movielens/1m/ |
| 規模 | ユーザー 6,040人 / 映画 3,883本 / レーティング 1,000,209件 |
| ライセンス | 研究目的での使用可（無償） |
| ドメイン | 映画推薦 |

#### User属性（`users.dat`）

| 属性 | 説明 | 値の例 |
|---|---|---|
| Gender | 性別 | M / F |
| Age | 年齢区分 | 1=Under 18, 18, 25, 35, 45, 50, 56 |
| Occupation | 職業（21カテゴリ） | 0=other, 1=academic/educator, 4=college/grad student, ... |
| Zip-code | 郵便番号 | 今回は除外 |

#### Item属性（`movies.dat`）

| 属性 | 説明 | 値の例 |
|---|---|---|
| Title | 映画タイトル（公開年含む） | "Toy Story (1995)" |
| Genres | ジャンル（パイプ区切り複数可） | "Animation\|Children's\|Comedy" |

---

### データセット 2：Yelp Open Dataset

| 項目 | 内容 |
|---|---|
| URL | https://www.yelp.com/dataset |
| 規模（参考） | ユーザー 200万人以上 / ビジネス 150万件以上 / レビュー 700万件以上 |
| ライセンス | 研究・教育目的での使用可（アカデミックライセンス） |
| ドメイン | 飲食店・店舗レビュー推薦 |
| 使用サブセット | レストランカテゴリのみ、かつレビュー数 ≥ 20件のユーザー・ビジネスに絞る |

#### User属性（`user.json`）

| 属性 | 説明 | 値の例 |
|---|---|---|
| review_count | 総レビュー数 | 42 |
| average_stars | ユーザーの平均評価 | 3.8 |
| yelping_since | アカウント登録年月 | "2010-05" → 「ベテランユーザー」等に変換 |
| elite | Elite認定年リスト（空なら非Elite） | [2019, 2020] |
| fans | フォロワー数 | 8 |

#### Item属性（`business.json`）

| 属性 | 説明 | 値の例 |
|---|---|---|
| name | ビジネス名 | "The Cheesecake Factory" |
| categories | カテゴリ（カンマ区切り） | "Restaurants, American (Traditional), Desserts" |
| stars | 総合評価 | 3.5 |
| attributes.RestaurantsPriceRange2 | 価格帯（1〜4） | "2" |
| city, state | 所在地 | "Las Vegas, NV" |

---

## 3. テキスト変換テンプレート

**言語：英語統一**（データが英語であり、日本語テンプレートにするとデータセットの選択肢が狭まるため）

### MovieLens 1M

#### User テキスト

```python
# Age区分のマッピング
AGE_MAP = {1: "under 18", 18: "18-24", 25: "25-34",
           35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}

# Occupation区分のマッピング（一部）
OCCUPATION_MAP = {
    0: "other", 1: "academic or educator", 2: "artist",
    4: "college or grad student", 7: "executive or managerial",
    12: "programmer", 17: "technician or engineer", ...
}

USER_TEXT_TEMPLATE = (
    "A {gender} user aged {age_group}, working as {occupation}."
)

# 例
# "A female user aged 25-34, working as a college or grad student."
```

#### Item テキスト

```python
ITEM_TEXT_TEMPLATE = (
    "Movie: {title}. Genres: {genres}."
)

# 例
# "Movie: Toy Story (1995). Genres: Animation, Children's, Comedy."
```

---

### Yelp Open Dataset

#### User テキスト

```python
# yelping_since を経験年数に変換
def tenure_label(yelping_since: str) -> str:
    years = 2024 - int(yelping_since[:4])
    if years >= 10: return "long-time"
    elif years >= 5: return "experienced"
    else: return "newer"

# Elite判定
def elite_label(elite: list) -> str:
    return "an Elite reviewer" if elite else "a regular user"

USER_TEXT_TEMPLATE = (
    "A {tenure} Yelp user with {review_count} reviews, "
    "averaging {average_stars:.1f} stars, and {elite_status}."
)

# 例
# "A long-time Yelp user with 152 reviews, averaging 3.8 stars, and an Elite reviewer."
```

#### Item テキスト

```python
PRICE_MAP = {"1": "inexpensive", "2": "moderate", "3": "pricey", "4": "upscale"}

ITEM_TEXT_TEMPLATE = (
    "{name} is a {price_range} restaurant in {city}, {state}. "
    "Categories: {categories}. Overall rating: {stars} stars."
)

# 例
# "The Cheesecake Factory is a moderate restaurant in Las Vegas, NV. "
# "Categories: American (Traditional), Desserts. Overall rating: 3.5 stars."
```

---

## 4. 検証するモデル一覧

本実験では以下の **7モデル** を比較する。
埋め込みモデルはすべて `intfloat/multilingual-e5-base`（frozen）を使用。
**item側のベクトル・インデックスは全モデルで共通・固定**。差異はクエリ側（ユーザー）の表現のみ。

| ID | モデル名 | カテゴリ | 説明 | 多様性の仕組み |
|---|---|---|---|---|
| M0 | **Single-Vector Baseline** | テキスト入力 | ユーザー属性テキストを1ベクトル化してANN検索 | なし（毎回同じ結果） |
| M1 | **Multi-Vector (Clustering)** | 多ベクトル | ユーザーの行動履歴をクラスタリングし、各クラスター代表ベクトルで別々にANN検索 | クラスター選択の確率的サンプリング |
| M2 | **Multi-Vector (MH-Attention)** | 多ベクトル | Multi-Head Attentionで複数の興味ベクトルを生成（ComiRec-SA型） | 推薦ごとに異なるヘッドベクトルを選択 |
| M3 | **Multi-Vector (Random Attr.)** | テキスト入力 | ユーザー属性のサブセットをランダムに選択しテキスト化、毎回異なるクエリベクトルを生成 | テキストのランダムサブセット化 |
| M4 | **Gaussian Noise Injection** | ノイズ注入 | M0のクエリベクトルにガウスノイズを加算し推論 | 埋め込み空間でのランダム摂動 |
| M5 | **MC Dropout** | ノイズ注入 | Encoderの推論時にDropoutを有効化し、毎回異なるクエリベクトルを生成 | Dropoutマスクの確率的変動 |
| M6 | **VAE-style Latent Sampling** | 確率的潜在変数 | mE5の出力に軽量VAEヘッドを乗せ、潜在変数zをサンプリングして推論 | 学習済み分布N(μ,σ²)からのサンプリング |

> **学習の要否によるグルーピング**：
M0〜M5は小速にitem embeddingから計算可能（**学習不要**またはは僅少なるfine-tuningのみ）。
M6は軽量VAEヘッドの学習が**必要**が、Validセットで学習するため完全に公平な比較が可能。

---

### 共通設定：埋め込みモデル

| 項目 | 内容 |
|---|---|
| モデル | `intfloat/multilingual-e5-base` |
| 次元数 | 768次元 |
| 入力プレフィックス | クエリ側: `"query: "` / アイテム側: `"passage: "` |
| プーリング | Average Pooling（attention maskで重み付け） |
| 正規化 | L2正規化（コサイン類似度で比較するため） |
| ファインチューニング | **なし（frozen）**：事前学習済みをそのまま使用 |

---

### M0: Single-Vector Baseline

```
ユーザー属性テキスト（全属性）
   → "query: " + text
   → multilingual-e5-base (frozen)
   → Average Pooling + L2 Norm
   → u ∈ ℝ^768（決定論的・毎回同一）
   → ANN検索（item側固定）→ Top-K
```

- **多様性なし**：同一ユーザーへの推薦は常に同じリスト
- 精度・多様性ともに他モデルの比較基準

---

### M1: Multi-Vector (Clustering)

PinnerSage（KDD 2020）型。ユーザーの行動履歴アイテムを階層クラスタリングし、各クラスターの代表アイテム（Medoid）のitem embeddingをクエリベクトルとして使用する。

```
ユーザーの高評価アイテム履歴
   → item embeddingを取得（train分のitem vectors）
   → 階層クラスタリング（Ward法、K=3〜5クラスター）
   → 各クラスターのMedoid（代表アイテム）を選出
   → K個のMedoidベクトル [m1, m2, ..., mK]

推薦時（確率的）：
   → シード値に基づきMedoidの中から1つをサンプリング
   → ANN検索（item側固定）→ Top-K
```

**多様性の仕組み**：シードを変えるたびに異なるMedoidが選ばれ、異なる候補が出る。

```python
# 推薦時の確率的サンプリング
rng = np.random.default_rng(seed)  # シードで制御
selected_medoid = rng.choice(medoids)  # K個から1つ選択
scores, indices = index.search(selected_medoid.reshape(1, -1), k=K)
```

---

### M2: Multi-Vector (Multi-Head Attention)

ComiRec-SA（KDD 2020）型。Multi-Head Attentionでユーザー興味を複数ベクトルに分解する。

```
ユーザーの高評価アイテム履歴 (N個のitem vectors)
   → 入力行列 X ∈ ℝ^{N×768}
   → Multi-Head Self-Attention（H=4〜8ヘッド）
   → H個の興味ベクトル [h1, h2, ..., hH] ∈ ℝ^{H×768}

推薦時（確率的）：
   → シードに基づきヘッドから1つをサンプリング
   → ANN検索（item側固定）→ Top-K
```

**多様性の仕組み**：各ヘッドが異なる興味の側面を表現しており、シードごとに異なるヘッドを使用することで多様な候補を取得。

```python
# Multi-Head Attention（概念的実装）
class MultiInterestEncoder(nn.Module):
    def __init__(self, d_model=768, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, item_history):  # (N, 768)
        # Q: 各ヘッドが独自の「興味ベクトル」に対応
        out, _ = self.attn(item_history, item_history, item_history)
        return out.mean(dim=0)  # (N_heads, 768) に近似的に対応
```

---

### M3: Multi-Vector (Random Attribute Subset)

最もシンプルな確率的クエリ変化。属性のサブセットをランダムに選んでテキストを生成し、推薦ごとに異なるクエリベクトルを作る。

```python
# MovieLensの例
ALL_ATTRS = ["gender", "age_group", "occupation"]

def generate_query_text(user, seed, min_attrs=1):
    rng = np.random.default_rng(seed)
    # 属性をランダムにサブセット選択
    n = rng.integers(min_attrs, len(ALL_ATTRS) + 1)
    selected = rng.choice(ALL_ATTRS, size=n, replace=False)
    parts = []
    if "gender" in selected:
        parts.append(f"A {user.gender} user")
    if "age_group" in selected:
        parts.append(f"aged {user.age_group}")
    if "occupation" in selected:
        parts.append(f"working as {user.occupation}")
    return "query: " + ", ".join(parts) + "."
```

**目的**：構造的な多ベクトル化（M1・M2）と比較するためのシンプルな確率的ベースライン。

---

### M4: Gaussian Noise Injection

M0のクエリベクトルにガウスノイズを直接加算する。学習不要で最も実装がシンプル。

```
M0のユーザークエリベクトル u ∈ ℝ^768（事前計算・固定）
   → ノイズ ε ∼ N(0, σ²I) をサンプリング（シードで制御）
   → ũ = u + ε
   → L2再正規化： ũ / ||ũ||
   → ANN検索（item側固定）→ Top-K
```

**多様性の仕組み**：シードごとに異なるノイズが加わり、クエリベクトルが微小に変化する。

```python
def noisy_query(user_vec: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, size=user_vec.shape)
    noisy = user_vec + noise
    return (noisy / np.linalg.norm(noisy)).astype(np.float32)

# ハイパーパラメーター：σ ∈ {0.01, 0.05, 0.10}でValid調整
```

**ノート**：σが小さいと関連性保持・多様性小、σが大きいと多様性大・精度低下のトレードオフ。最適σをValidで調整する。

---

### M5: MC Dropout

推論時にDropoutを有効にみたMonte Carlo Dropout。Gal & Ghahramani（2016）が科学的基盤。

```
ユーザー属性テキスト
   → "query: " + text
   → multilingual-e5-base (frozen) + Dropout層（p=0.1。0.2）
      ※ 推論時も model.train() 相当の状態にしDropoutをON
   → Average Pooling + L2 Norm
   → ũ ∈ ℝ^768（シードごとに異なるDropoutマスク）
   → ANN検索（item側固定）→ Top-K
```

**多様性の仕組み**：Dropoutマスクのランダム性により、推論ごとに微小に異なる埋め込みベクトルが生成される。

```python
class MCDropoutEncoder(nn.Module):
    def __init__(self, encoder, dropout_rate=0.1):
        super().__init__()
        self.encoder = encoder     # mE5 frozen
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_ids, attention_mask):
        # 推論時も self.training=True 相当でDropoutを有効化
        with torch.no_grad():
            hidden = self.encoder(input_ids, attention_mask).last_hidden_state
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1) / \
                 attention_mask.sum(1, keepdim=True)
        dropped = self.dropout(pooled)   # 推論時もDropout ON
        return F.normalize(dropped, dim=-1)

# ハイパーパラメーター： dropout_rate ∈ {0.05, 0.10, 0.20}でValid調整
```

**M4との違い**：M4は埋め込みベクトル全体に同程度のノイズ、M5はEncoder内部のニューロンごとにランダムに山落ちする（より構造的なノイズ）。

---

### M6: VAE-style Latent Sampling

**本実験で最も理論的に整備されたランダム性注入**。Liang et al. (2018) MultVAEをベースに、mE5の出力の上に軽量VAEヘッドを乗せる。

#### アーキテクチャ

```
ユーザー属性テキスト
   → multilingual-e5-base (frozen) + Avg Pooling
   → v ∈ ℝ^768（固定決定論的）
         ↓ 軽量VAE Encoder Head（学習可能、MLP）
         ↓               ↓
        μ ∈ ℝ^d      logσ² ∈ ℝ^d
              ↓ 再パラメータリゼーション
        z = μ + σ · ε（ε ∼ N(0,I)、シードで制御）
         ↓ L2正規化
   → ANN検索（item側固定）→ Top-K
```

#### VAEの損失関数

$$\mathcal{L} = \underbrace{\|v - \hat{v}\|^2}_{\text{reconstruction}} + \beta \cdot \underbrace{\text{KL}(q(z|v) \| \mathcal{N}(0,I))}_{\text{regularization}}$$

- **Reconstruction**：zからvを再構成できるか（Decoder Headで実装）
- **KL正則化**：zの分布をN(0,I)に近づける → 推論時に任意のεでサンプリング可能になる
- **β**（ハイパーパラメーター）：大きいと多様性大／精度小、小さいとその逆

```python
class VAEHead(nn.Module):
    """軽量VAEヘッド（mE5出力768dim上に乗せる）"""
    def __init__(self, input_dim=768, latent_dim=128):
        super().__init__()
        # Encoder Head
        self.fc_mu    = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        # Decoder Head（学習時のreconstructionlossのみ使用）
        self.decoder  = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu, logvar, seed=None):
        std = torch.exp(0.5 * logvar)
        if seed is not None:
            torch.manual_seed(seed)
        eps = torch.randn_like(std)  # シードで制御
        return mu + std * eps

    def forward(self, x, seed=None):
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar, seed)
        recon = self.decoder(z)
        return z, mu, logvar, recon

def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

# ハイパーパラメーター：
# latent_dim ∈ {64, 128, 256}、β ∈ {0.5, 1.0, 2.0}でValid調整
```

**付記事項**：推論時はDecoderは不要。`z = mu + sigma * eps`をクエリベクトルとしてANN検索するのみ。latent_dim≠768の場合、プロジェクション層でANN空間の次元に切り出すか、z後にリニア層を挿入する。

---

## 5. ANNインデックス

```python
# アイテムベクトルを事前にインデックス化（item側固定・全モデル共通）
import faiss

index = faiss.IndexFlatIP(768)  # 内積（＝コサイン類似度、L2正規化済み）
index.add(item_vectors)  # shape: (N_items, 768)
# ※ このインデックスはM0〜M6で共用し、変更しない

# 検索
scores, indices = index.search(query_vector, k=K)  # Top-K検索
# (注意) M6のVAEはlatent_dim ≠ 768の場合、
# クエリ側にリニア層を挿入し768次元に展開する（またはindex側も小次元化）
```

---

## 6. 評価方法

### Ratingしきい値の決定と根拠

**両データセットともに rating ≥ 4 を「正解（正例）」として採用する。**

#### 決定根拠

関連研究における使用実績に基づく：

| 論文 | データセット | しきい値 | 備考 |
|---|---|---|---|
| He et al. (2017) Neural Collaborative Filtering (WWW) | MovieLens-1M | **≥ 4** | NCF公式実装の設定値 |
| Rendle et al. (2009) BPR (UAI) | MovieLens系 | **≥ 4** | BPR論文での標準設定 |
| Li et al. (2019) MIND (CIKM) | Amazon, Tmall | ≥ 4相当 | 購入 / 高評価をpositive扱い |
| Cen et al. (2020) ComiRec (KDD) | Amazon, Taobao | ≥ 4相当 | 同上 |
| Yelp系推薦研究（多数） | Yelp | **≥ 4** | 4以上を「推薦に値する」と判断 |

#### ≥ 4 を選んだ理由

1. **高精度のポジティブシグナル**：5段階評価において4・5は「満足 / 強く満足」を示し、推薦に値するアイテムの定義として適切
2. **先行研究との比較可能性**：NCF・BPR等の主要ベースラインと同じ設定を使うことで、精度の絶対値を既存報告と照合できる
3. **データ密度のバランス**：MovieLens 1M では rating ≥ 4 が約55%を占め、正例が少なすぎず多すぎない適切な密度になる（≥ 5 だと約22%に激減しスパースになりすぎる）

#### ≥ 3 を選ばない理由

- 3は「普通・可もなく不可もない」という中立的な評価であり、「推薦したい」という積極的な好みではない
- 3を含めるとノイズが増え、Two-Tower Model が「本当に好きなアイテム」を学習しにくくなる

---

### データ分割

**時系列に基づく Train : Valid : Test = 8 : 1 : 1 分割**

| 分割 | 割合 | 用途 |
|---|---|---|
| Train | 80%（時系列前半） | M1のクラスタリング・M2のAttention学習で使用する行動履歴。M0・M3は不使用（frozen推論のみ） |
| Valid | 10%（時系列中盤） | ハイパーパラメーター調整（クラスター数K・ヘッド数H・しきい値等） |
| Test | 10%（時系列後半） | 最終メトリクス報告に使用 |

```python
# 分割ロジック（ユーザーごとに時系列で分割）
for user_id, interactions in user_interactions.items():
    interactions = sorted(interactions, key=lambda x: x.timestamp)
    n = len(interactions)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)
    train[user_id] = interactions[:train_end]
    valid[user_id] = interactions[train_end:valid_end]
    test[user_id]  = interactions[valid_end:]

# rating ≥ 4 のみを正解アイテムとして使用
test_ground_truth[user_id] = [
    x.item_id for x in test[user_id] if x.rating >= 4
]

### 6.1 精度指標：Recall@K（複数回推薦の累積）

```
複数回の推薦結果を蓄積したときの Recall を評価

Recall@K (累積) = |推薦結果の和集合 ∩ 正解アイテム| / |正解アイテム|
```

- K = 10, 20, 50 で計測
- 推薦回数 T = 1, 3, 5, 10 回で変化を観察

```python
# 疑似コード
for trial in range(T):
    recommended_items_t = ann_search(user_vector, k=K)
    all_recommended.update(recommended_items_t)

recall = len(all_recommended & ground_truth) / len(ground_truth)
```

### 6.2 多様性指標：推薦間重複率（Temporal Overlap Rate）

本実験の核心となる指標。「推薦するたびに候補が変わるか」を測定する。

```
Overlap Rate between trial t1 and t2 = |R(t1) ∩ R(t2)| / K
```

- K = 10 件推薦した場合に、別の試行での推薦リストと何件重複するか
- 重複率 = 1.0 → 完全に同じリスト（多様性なし）
- 重複率 = 0.0 → 全く異なるリスト

```python
# 全ての試行ペアの平均重複率
overlap_rates = []
for t1, t2 in combinations(range(N_TRIALS), 2):  # N_TRIALS=5
    overlap = len(set(R[t1]) & set(R[t2])) / K
    overlap_rates.append(overlap)
mean_overlap = mean(overlap_rates)
```

> **M0（Single-Vector Baseline）**：Frozen + 決定論的ANN のため重複率は常に **1.0**。これを「多様性なし状態」のベースラインとして記録する。

### 6.3 補助指標

| 指標 | 定義 |
|---|---|
| Hit Rate@K | 推薦K件中に正解が1件以上含まれる割合 |
| NDCG@K | Normalized Discounted Cumulative Gain（順位を考慮した精度） |
| ILD (Intra-List Diversity) | 推薦リスト内のアイテム間の平均コサイン距離 |
| Coverage | 推薦されたユニークアイテム数 / 全アイテム数 |

---

### 6.4 実験の繰り返し：5シード平均

確率的な要素を含むモデル（M1・M2・M3）の結果は、**別シードで5回実施し平均・標準偏差を報告する**。

#### シード設定

```python
SEEDS = [0, 1, 2, 3, 4]  # 5回の試行に使用するシード
```

#### シードが影響する箇所

| モデル | シードの影響箇所 | 学習要否 |
|---|---|---|
| M0 | シード不要（決定論的） | 不要 |
| M1 | 推薦時のMedoidサンプリング | 不要（クラスタリングのみ） |
| M2 | 推薦時のAttentionヘッドサンプリング | 不要（Attentionのみ） |
| M3 | 属性サブセットのランダム選択 | 不要 |
| M4 | ノイズサンプリング ε ∼ N(0,σ²) | 不要 |
| M5 | Dropoutマスクのランダム化 | 不要 |
| M6 | VAEの再パラメータリゼーション ε ∼ N(0,I) | **必要**（VAEヘッドのみ） |

#### 結果の報告形式

```python
# 各シードで評価を実行
results_per_seed = []
for seed in SEEDS:
    metrics = evaluate_model(model, test_data, seed=seed)
    results_per_seed.append(metrics)

# 平均・標準偏差を計算して報告
report = {
    "Recall@10":    f"{mean(r['recall@10']  for r in results_per_seed):.4f} ± "
                    f"{std(r['recall@10']   for r in results_per_seed):.4f}",
    "Overlap Rate": f"{mean(r['overlap']    for r in results_per_seed):.4f} ± "
                    f"{std(r['overlap']     for r in results_per_seed):.4f}",
    ...
}
```

#### 最終報告テーブルのフォーマット（例）

| Model | カテゴリ | Recall@10 ↑ | NDCG@10 ↑ | Overlap Rate ↓ | ILD ↑ |
|---|---|---|---|---|---|
| M0 Single-Vector | ベースライン | x.xxxx | x.xxxx | **1.0000** | x.xxxx |
| M1 Clustering | 多ベクトル | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx |
| M2 MH-Attention | 多ベクトル | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx |
| M3 Random Attr. | テキスト入力 | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx |
| M4 Gaussian Noise | ノイズ | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx |
| M5 MC Dropout | ノイズ | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx |
| M6 VAE Sampling | VAE | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx | x.xxxx ± x.xxxx |

---

## 7. 実験手順

```
Step 1: データ準備
  ├── [MovieLens] users.dat / movies.dat / ratings.dat のダウンロード・読み込み
  ├── [Yelp] user.json / business.json / review.json のダウンロード・読み込み
  │          （レストランカテゴリのみ、レビュー数≥20のユーザー・ビジネスに絞る）
  ├── 時系列ソート後に Train:Valid:Test = 8:1:1 で分割
  └── 英語テキストテンプレートに変換

Step 2: 埋め込み生成（multilingual-e5-base、frozen）
  ├── 全アイテムの属性テキスト → item_vectors → 保存（全モデル共用）
  ├── [M0] 全ユーザーの全属性テキスト → user_vectors（決定論的・事前計算）
  ├── [M1] Trainの行動履歴からクラスタリング → Medoidベクトルを保存
  ├── [M2] Trainの行動履歴でMulti-Head Attentionを事前計算 → ヘッドベクトルを保存
  ├── [M3] 属性サブセットのテキスト生成ロジックを実装
  ├── [M4] M0のuser_vectorsをそのまま流用（推論時にノイズを加算）
  ├── [M5] MCDropoutEncoderをラップしてuser_vectorsを生成
  └── [M6] VAEHeadをValidセットで学習 → μ・σパラメーターを保存

Step 3: ANNインデックス構築（全モデル共通）
  └── item_vectors を FAISS IndexFlatIP に追加（以降変更しない）

Step 4: Validで各モデルのハイパーパラメーター調整
  ├── [M1] クラスター数 K ∈ {3, 4, 5}
  ├── [M2] Attentionヘッド数 H ∈ {4, 8}
  ├── [M3] 最小属性選択数 min_attrs ∈ {1, 2}
  ├── [M4] ノイズ強度 σ ∈ {0.01, 0.05, 0.10}
  ├── [M5] Dropout率 p ∈ {0.05, 0.10, 0.20}
  └── [M6] VAEHead学習（latent_dim ∈ {64,128}、β ∈ {0.5,1.0,2.0}）→ 最良モデルを保存

Step 5: Testで最終評価（5シード × 7モデル）
  ├── SEEDS = [0, 1, 2, 3, 4]
  ├── 各シードで M0〜M6 の全評価指標を計算
  │     ├── Recall@K (K=10,20,50)
  │     ├── Hit Rate@K / NDCG@K
  │     └── Temporal Overlap Rate / ILD / Coverage
  └── 5シードの平均・標準偏差を集計（M0は決定論的なので固定値）

Step 6: 結果の記録と可視化
  ├── metrics.json に全結果を保存
  ├── 精度 vs 多様性のトレードオフ散布図を生成
  ├── ユーザーセグメント別分析（性別、年齢、職業 / Yelp: Elite有無）
  └── result/plan_001/{movielens,yelp}/ に保存
```

---

## 8. 期待される結果

### 精度について

- Frozen mE5の埋め込みは属性テキストの**意味的類似度**を捉えるため、協調フィルタリングと異なる特性が出る
- 「同じジャンル・年代の映画」「同じ属性のユーザー」が近い空間にマッピングされる
- ランダムベースラインを大幅に上回ることを期待

### 多様性について（ベースライン確認）

| 条件 | 期待される重複率 |
|---|---|
| 決定論的ANN（ベースライン） | **1.0**（毎回同じ結果） |
| 多ベクトル化後（plan_002） | 0.3〜0.7 |
| MMR適用後（plan_003） | 0.1〜0.4 |

---

## 9. 後続実験との接続

本実験（plan_001）は以下の実験の比較ベースラインとなる：

| 実験ID | 内容 |
|---|---|
| plan_001 | 本実験：属性テキスト Two-Tower ベースライン |
| plan_002（予定） | クエリ側多ベクトル化（PinnerSage型クラスタリング） |
| plan_003（予定） | クエリ側多ベクトル化（ComiRec-SA型 Multi-Head Attention） |
| plan_004（予定） | MMR によるポスト検索多様化 |
| plan_005（予定） | DPP によるポスト検索多様化 |

---

## 10. 環境・依存ライブラリ

```python
# requirements
torch>=2.0
transformers>=4.35
sentence-transformers>=2.2
faiss-cpu>=1.7      # または faiss-gpu
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3   # M1クラスタリング（AgglomerativeClustering）
tqdm
# M6 VAE学習用（追加）
torch>=2.0          # 上記と共通
# （追加ライブラリなし。VAEHeadはPyTorch nn.Moduleで実装）
```

---

## 11. ディレクトリ構成（予定）

```
tt-embedding-variation/
├── article/              # 先行研究まとめ
├── experiment_plan/      # 実験計画
│   └── plan_001.md      # 本ファイル
├── src/                  # 実装コード（実験時に作成）
│   ├── data/
│   │   ├── preprocess_movielens.py
│   │   └── preprocess_yelp.py
│   ├── model/
│   │   └── two_tower.py
│   └── evaluate/
│       ├── recall.py
│       └── diversity.py
└── result/               # 実験結果（実験時に作成）
    └── plan_001/
        ├── movielens/
        │   ├── metrics.json
        │   └── figures/
        └── yelp/
            ├── metrics.json
            └── figures/
```

---

## 12. 決定済み事項・残課題

### ✅ 決定済み

| 項目 | 決定内容 |
|---|---|
| テキストテンプレートの言語 | **英語**（データと一致させ、将来のデータセット拡張にも対応） |
| データセット | **MovieLens 1M** + **Yelp Open Dataset** |
| Ratingしきい値 | **≥ 4**（NCF/BPR等の先行研究に準拠、§6参照） |

### ⬜ 残課題

- [ ] テストユーザー数を全員とするか、一部サンプリングするか（計算コスト確認後に決定）
- [ ] Yelpのサブセット絞り込み条件の最終調整（レビュー数しきい値等）
- [ ] mE5-base と mE5-large の比較実験を plan_001 の範囲でやるか plan_002 に持ち越すか
