# MMR: The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries

## 基本情報

| 項目 | 内容 |
|---|---|
| タイトル | The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries |
| 著者 | Jaime Carbonell, Jade Goldstein |
| 所属 | Carnegie Mellon University |
| 会議 | SIGIR 1998 |
| URL | [CMU論文ページ](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf) |
| 発表年 | 1998 |

## 概要

推薦・情報検索における多様性問題の**古典的かつ最も広く使われる解法**。
クエリへの関連性と、既に選択済みのアイテムとの多様性を線形結合した「Maximal Marginal Relevance（MMR）」基準でアイテムを逐次選択する。
シンプルながら効果的で、現在でも産業界・学術界で広く使用されている。

## 問題設定

- 通常のランキング：関連性スコアの高い順に並べるだけ → 上位結果が互いに類似しがち
- ユーザーが求めているのは「関連性が高く、かつ重複が少ない」多様なリスト
- ドキュメント検索・推薦・文書要約において、関連性と多様性のトレードオフが課題

## MMRアルゴリズム

### 定義

$$\text{MMR} = \arg\max_{D_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}_1(D_i, Q) - (1 - \lambda) \cdot \max_{D_j \in S} \text{Sim}_2(D_i, D_j) \right]$$

| 記号 | 意味 |
|---|---|
| $Q$ | クエリ（ユーザーベクトル） |
| $R$ | 候補アイテム全体 |
| $S$ | 既に選択済みのアイテム集合 |
| $\text{Sim}_1(D_i, Q)$ | アイテム $D_i$ とクエリ $Q$ の類似度（関連性） |
| $\text{Sim}_2(D_i, D_j)$ | アイテム間の類似度 |
| $\lambda$ | 関連性と多様性のトレードオフパラメーター |

### アルゴリズム（逐次選択）

```python
S = []  # 選択済みリスト
R = [全候補アイテム]

while len(S) < N:  # N件選択するまで繰り返し
    next_item = argmax over (D_i in R - S):
        lambda * Sim1(D_i, query) - (1 - lambda) * max(Sim2(D_i, D_j) for D_j in S)
    S.append(next_item)
```

### λ の意味

| λ 値 | 動作 |
|---|---|
| λ = 1.0 | 純粋な関連性ランキング（多様性なし） |
| λ = 0.0 | 純粋な多様性（関連性無視） |
| 0 < λ < 1 | 関連性と多様性のバランス |

## 本実験との関連性

### ✅ 適用可能な点

- **item側のベクトルを変えない**：MMRはANN検索後のポスト処理として機能し、item embeddingそのものは変更不要
- Two-Tower ModelのANN検索結果に対してそのまま適用できる
- λ を調整することで精度（関連性）を保ちながら多様性を制御できる
- 実装がシンプルで計算コストが低い

### ✅ Two-Tower Modelとの組み合わせ方

```
1. Two-Tower Model でクエリベクトル生成
2. ANNインデックス（item側固定）で Top-M 候補取得（M > N）
3. MMR で Top-M からN件を選択（関連性と多様性を両立）
4. 最終推薦リスト（N件）を返す
```

### ✅ 「推薦ごとに候補を変える」多様性との関係

- MMRはリスト内多様性（intra-list diversity）を保証するが、**推薦実行ごとの変化（temporal diversity）**は保証しない
- ただし、Two-Tower Modelのクエリ側に確率的な要素を追加した上でMMRを後段に置くと、両方の多様性を実現できる

### ⚠️ 注意点

- 逐次選択のためO(N²)の計算量がかかる（候補数が多い場合は要注意）
- λ の最適値はドメインやユーザーによって異なり、チューニングが必要
- 「多様性」の定義がコサイン類似度ベースであり、意味的な多様性とは若干異なる場合がある

## 実用例

- **Google, Amazon, Netflix** 等の大手推薦システムで実際に使用
- Elasticsearch/OpenSearchにMMR実装が組み込み済み
- LangChainのVector Store検索でもMMRが標準オプションとして提供

## 引用

```bibtex
@inproceedings{carbonell1998mmr,
  title={The use of MMR, diversity-based reranking for reordering documents and producing summaries},
  author={Carbonell, Jaime and Goldstein, Jade},
  booktitle={Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval},
  pages={335--336},
  year={1998}
}
```
