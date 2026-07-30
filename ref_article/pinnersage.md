# PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest

## 基本情報

| 項目 | 内容 |
|---|---|
| タイトル | PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest |
| 著者 | Aditya Pal, Chantat Eksombatchai, Yitong Zhou, Bo Zhao, Charles Rosenberg, Jure Leskovec |
| 所属 | Pinterest |
| 会議 | KDD 2020 |
| arxiv | [2007.03634](https://arxiv.org/abs/2007.03634) |
| 発表年 | 2020 |

## 概要

Pinterestが本番環境でデプロイした**マルチモーダルユーザー埋め込みフレームワーク**。
ユーザーの行動（ピン保存・クリック）を階層的クラスタリングで複数クラスターに分割し、各クラスターの代表ピン（Medoid）をユーザーの「興味ベクトル」として使用する。
単一埋め込みと比較して大幅な性能向上を達成し、Pinterest全体の推薦システムで稼働中。

## 問題設定

- 単一の高次元ベクトルでユーザーを表現 → ユーザーの多様な興味を1つに圧縮してしまう
- PinterestではユーザーがFashion/Food/Travel等の全く異なるカテゴリに同時に興味を持つ
- 単一ベクトルは「平均的な興味」を表すため、どのカテゴリの推薦も中途半端になる

## 提案手法：PinnerSage

### アーキテクチャ概要

```
ユーザーの行動履歴（ピン保存・クリック）
     │
     ▼
階層的クラスタリング（Ward法）
     │ 意味的に近い行動をグループ化
     ▼
各クラスターの Medoid（代表ピン）抽出
     │ 解釈可能なユーザー表現
     ▼
複数のMedoidベクトル = ユーザー表現
     │
     ▼
各Medoidベクトルでページ推薦（ANN検索）
```

### 1. 階層的クラスタリング（Ward法）

- ユーザーの行動履歴を意味的に似たピンのクループに分割
- Ward法：クラスター内分散を最小化するようにマージ
- 結果として、ユーザーの各「興味トピック」が1つのクラスターになる

### 2. Medoid（代表ピン）の使用

- 各クラスターの「重心」ではなく、**実際のピン（Medoid）をその興味の代表**として使用
- 理由：クラスター重心は既存のアイテムに対応しないため、ANNインデックスで直接検索できない
- **Medoidを用いることで、既存のitem ANNインデックスをそのまま使用可能**

### 3. リアルタイム更新

- **日次バッチ処理**：長期的な行動履歴のクラスタリングを更新
- **オンライン更新**：直近の行動（最新のピン保存等）をリアルタイムで反映

### 4. 推薦時の動作

```
ユーザー → 複数Medoidベクトル [m1, m2, ..., mK]
各 mi でANNインデックス（item側固定）を検索
→ K個の候補集合を結合
→ ランキングへ
```

## 本実験との関連性

### ✅ 適用可能な点

- **item側のベクトルを全く変更しない**：MedoidはItemのembeddingをそのまま使うため、既存のANNインデックスを流用
- クエリ側（ユーザー）を複数ベクトルで表現することで推薦多様性が自然に実現
- 毎回使用するMedoidを選択する際に確率的な選択を加えれば、**推薦ごとに異なる候補**を取得可能

### ✅ 言語情報埋め込みとの親和性

- 本実験では言語情報を元にitem embeddingが作られているが、PinnerSageでは**pin embeddingがそのままユーザー興味ベクトル（Medoid）として使える**
- 言語埋め込み空間において意味的に異なるアイテムを中心としたクラスターを形成できれば、同じパラダイムが適用可能

### ⚠️ 注意点

- クラスタリングをオフラインで実行するため、リアルタイム性に限界がある
- 行動履歴が少ないユーザー（コールドスタート）では有効なクラスターを形成できない
- 論文では精度向上が主目的であり、「推薦ごとに候補を変える」多様性は明示的に議論されていない

## 実験結果

- オフライン実験（Hit Rate）：単一埋め込みベースラインを大幅に上回る
- オンラインA/Bテスト：本番Pinterestで有意なエンゲージメント向上を確認
- 本番デプロイ済み（Pinterest全サービス）

## 引用

```bibtex
@inproceedings{pal2020pinnersage,
  title={PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest},
  author={Pal, Aditya and Eksombatchai, Chantat and Zhou, Yitong and Zhao, Bo and Rosenberg, Charles and Leskovec, Jure},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={2311--2320},
  year={2020}
}
```
