# AMER: Beyond Single Embeddings: Capturing Diverse Targets with Multi-Query Retrieval

## 基本情報

| 項目 | 内容 |
|---|---|
| タイトル | Beyond Single Embeddings: Capturing Diverse Targets with Multi-Query Retrieval |
| 著者 | Hung-Ting Chen et al. |
| 会議 | arxiv (preprint) |
| arxiv | [2511.02770](https://arxiv.org/abs/2511.02770) |
| 発表年 | 2024 |

## 概要

情報検索・推薦における**単一クエリベクトルの限界**を定量化し、自己回帰的に複数クエリベクトルを生成する新しいRetrieverアーキテクチャ「AMER（Autoregressive Multi-Embedding Retriever）」を提案。
複数の検索ターゲット（文書/アイテム）の分布がマルチモーダルな場合に、単一ベクトルでは捉えきれない問題を根本から解決する。

## 問題設定

### 単一埋め込みの限界（定量的分析）

- 既存の全てのRetrieverは**1つのクエリベクトル**から近傍検索を行う
- クエリに対して「関連文書/アイテムの分布」がマルチモーダル（複数の意味的に離れた集団）の場合、1つのベクトルでは全て捉えられない
- 実験で確認：**ターゲット埋め込み間の距離が大きいほど、単一ベクトルRetrieverの性能が低下**

### 具体的な問題例

```
クエリ：「Pythonで機械学習」
→ 関連文書1：PyTorchの使い方（Deep Learning系）
→ 関連文書2：scikit-learnの解説（Traditional ML系）
→ 関連文書3：データ前処理の方法（Data Engineering系）

これら3つは意味的に離れており、1つのベクトルで全て捉えるのは困難
```

## 提案手法：AMER

### アーキテクチャ

```
クエリ入力
    │
    ▼
Transformer Encoder（クエリ表現）
    │
    ▼
自己回帰的多クエリ生成
    z1 → z2（z1を条件として生成） → ... → zK
    │
    ▼
各 zi で文書コーパスを独立にANN検索
    │
    ▼
K個の候補集合を結合して最終結果
```

### 自己回帰的生成の仕組み

```
z1 = Encoder(query)
z2 = Decoder(query, z1)     ← z1と異なるベクトルを強制
z3 = Decoder(query, z1, z2) ← z1, z2と異なるベクトルを強制
...
```

- 各ステップで過去のベクトルを条件として次のベクトルを生成
- これにより、**意味的に異なる複数のクエリベクトル**が自然に生成される

### 訓練のための Hungarian Matching

- 複数のクエリベクトルと複数のターゲット文書を1対1で対応付ける必要がある
- **ハンガリアンアルゴリズム**でコスト最小の対応付けを動的に決定して訓練

## 実験結果

### 合成データ実験

- 単一埋め込みBaselineと比較して**4倍の性能向上**（多様なターゲット分布に対して）

### 実世界データ実験

| データセット | 単一埋め込み | AMER | 相対改善 |
|---|---|---|---|
| Dataset 1 | baseline | AMER | +4% |
| Dataset 2 | baseline | AMER | +21% |

- **ターゲット間の類似度が低いデータほど改善幅が大きい**

## 本実験との関連性

### ✅ 適用可能な点

- **item側（文書コーパス）のベクトルは一切変更しない**：AMERは純粋にクエリ側で複数ベクトルを生成
- 生成される複数クエリベクトルは互いに「異なる方向」を向くように設計されており、多様な候補を自然に取得できる
- 言語情報をベースにしたembeddingとの親和性が高い（TransformerベースのEncoder）

### ✅ 本実験の「毎回異なる候補」への応用

- 推薦実行ごとにどのクエリベクトルを使用するかを確率的に選択することで、temporal diversityを実現できる
- または、自己回帰的に生成されたベクトルのうち毎回異なるものを使用する戦略も可能

### ⚠️ 注意点

- 情報検索（テキスト検索）で評価されており、推薦システムへの直接適用には追加検討が必要
- 自己回帰的生成は推論時の計算コストが上がる可能性がある
- preprint段階であり、査読済み論文ではない

## 引用

```bibtex
@article{chen2024amer,
  title={Beyond Single Embeddings: Capturing Diverse Targets with Multi-Query Retrieval},
  author={Chen, Hung-Ting and others},
  journal={arXiv preprint arXiv:2511.02770},
  year={2024}
}
```
