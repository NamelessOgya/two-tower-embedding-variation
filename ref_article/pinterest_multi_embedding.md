# Synergizing Implicit and Explicit User Interests: A Multi-Embedding Retrieval Framework at Pinterest

## 基本情報

| 項目 | 内容 |
|---|---|
| タイトル | Synergizing Implicit and Explicit User Interests: A Multi-Embedding Retrieval Framework at Pinterest |
| 著者 | Pinterest Research Team |
| 所属 | Pinterest |
| 会議 | KDD 2025 |
| arxiv | [2506.23060](https://arxiv.org/abs/2506.23060) |
| 発表年 | 2025 |

## 概要

PinterestのHome Feedに本番デプロイされた最新の多埋め込み検索フレームワーク。
**暗黙的（Implicit）** なユーザー興味と **明示的（Explicit）** なユーザー興味を組み合わせて、複数のユーザー埋め込みベクトルを生成する。
多様なユーザー興味（特にロングテール）をカバーすることでフィードの多様性と品質を同時に向上させた。

## 問題設定

### 従来のTwo-Towerモデルの限界

- 従来のTwo-Towerモデルは**ユーザーを1つの埋め込みベクトル**で表現
- 課題1：ユーザーとアイテムの特徴間のインタラクションが限定的
- 課題2：主要な使用事例（上位の興味）に偏りやすく、**ロングテール・多様な興味を捉えられない**
- 結果：フィードが均質になりエンゲージメントの多様性が失われる

## 提案手法

### システム概要

```
ユーザー
  ├─ 行動履歴（Implicit）→ DCM → 複数の暗黙的興味ベクトル
  └─ フォロートピック（Explicit）→ CR → 複数の明示的興味ベクトル
           │
           ▼
     多ベクトルANN検索（item側固定）
           │
           ▼
   ラウンドロビン結合 → ランキング → ブレンド
```

### 1. Differentiable Clustering Module (DCM)：暗黙的興味

**MINDのカプセルルーティングを改良した新手法**

```
ユーザーの行動履歴（エンゲージしたピン）
       │
       ▼
Validity-Aware Farthest Point Initialization (VA-FPI)
  ・ランダム初期化 → 多様性が低いクラスター
  ・VA-FPI：互いに最も遠い点を初期セントロイドとして選択
  ・クラスター間の多様性を最初から保証
       │
       ▼
Single-Assignment Routing（単一割り当て）
  ・各履歴アイテムは1つのクラスターのみに属する
  ・クラスターの明確な分離を保証
       │
       ▼
K個の暗黙的興味ベクトル
```

**従来のカプセルルーティングとの違い：**
- 通常のDynamic Routing：ソフト割り当て（各アイテムが複数クラスターに重み付きで属する）
- DCM：ハード割り当て（各アイテムは1クラスターのみ） → より明確な興味分離

### 2. Conditional Retrieval (CR)：明示的興味

```
ユーザーが明示的にフォローしたトピック（例：料理、旅行、ファッション）
       │
       ▼
各トピックをコンテキストとしてユーザー表現を条件付け
       │
       ▼
トピックごとの興味ベクトル生成
```

- ユーザーが明示的に「好き」と表現した興味を直接エンコード
- 暗黙的興味（行動履歴ベース）では見落とされがちな興味を補完

### 3. 検索・結合戦略

```
DCM興味ベクトル [d1, d2, ..., dK1] → ANN検索 → 候補集合D
CR興味ベクトル  [c1, c2, ..., cK2] → ANN検索 → 候補集合C
         │
         ▼
ラウンドロビン結合（D1, C1, D2, C2, ...）
- 暗黙・明示の双方から均等にサンプリング
         │
         ▼
ランキングステージへ
```

## 本実験との関連性

### ✅ 最も関連性が高い先行研究

- **item側のベクトルを変更しない**：ANNインデックス（item側）は固定、クエリ側のみ変化
- **推薦多様性を明示的な目標として設定**：論文中でフィードの多様性指標の改善を評価
- 言語情報埋め込みとの親和性：DCMはユーザー行動の言語的な意味空間でのクラスタリングに相当
- **本番デプロイ済みで効果が確認済み**（Pinterestホームフィード）

### ✅ VA-FPIの重要な示唆

- クラスター初期化を「多様性最大化」の観点で行うことで、生成される興味ベクトルが互いに離れた方向を向く
- これを本実験に適用すれば：言語埋め込み空間でユーザー興味を多様なクラスターに分割することで、毎回異なる方向のクエリベクトルを取得できる

### ✅ Implicit + Explicit の相補的アプローチ

- 本実験でもユーザーの明示的な好み（プロファイル情報）と暗黙的な行動履歴を組み合わせることができる

### ⚠️ 注意点

- DCMとCRを組み合わせるシステム設計は複雑
- フォロートピック等の明示的なシグナルが必要（データによっては取得できない場合がある）

## 実験結果

- **ユーザーエンゲージメント指標**：有意な向上（A/Bテスト）
- **フィード多様性指標**：有意な向上（A/Bテスト）
- PinterestホームフィードにデプロイされA/Bテストで確認済み

## 引用

```bibtex
@inproceedings{pinterest2025multiembedding,
  title={Synergizing Implicit and Explicit User Interests: A Multi-Embedding Retrieval Framework at Pinterest},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025},
  note={arXiv:2506.23060}
}
```
