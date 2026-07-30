# ComiRec: Controllable Multi-Interest Framework for Recommendation

## 基本情報

| 項目 | 内容 |
|---|---|
| タイトル | Controllable Multi-Interest Framework for Recommendation |
| 著者 | Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, Jie Tang |
| 所属 | Alibaba Group, Tsinghua University |
| 会議 | KDD 2020 |
| arxiv | [2005.09347](https://arxiv.org/abs/2005.09347) |
| 発表年 | 2020 |

## 概要

MINDを発展させた**制御可能な多興味フレームワーク（ComiRec）**。
複数の興味ベクトルによる候補取得に加えて、**精度と多様性のトレードオフを制御するパラメーター**を明示的に設けた点が新しい。
AmazonとTaobaoの実験でSOTAを達成し、Alibabaのオフラインクラスタにデプロイ済み。

## 問題設定

- ユーザーの行動シーケンスから「次に興味を持つアイテム」を予測する逐次推薦（Sequential Recommendation）の問題を設定
- 従来手法：ユーザー全体を1つの埋め込みで表現 → 多様な興味を捉えられない
- MIND等の先行手法：多興味は表現できるが、**精度と多様性のバランスが制御できない**

## 提案手法：ComiRec

### アーキテクチャ概要

```
ユーザー行動シーケンス
     │
     ▼
Multi-Interest Module（多興味モジュール）
     │ K個の興味ベクトル
     ▼
ANN検索（各興味ベクトルで別々に検索）
     │ 候補集合の結合
     ▼
Aggregation Module（集約モジュール）
     │ 制御因子 λ によるトレードオフ
     ▼
最終推薦リスト
```

### 1. Multi-Interest Module（2種類）

**ComiRec-DR（Dynamic Routing版）**
- MINDと同じカプセルネットワークのDynamic Routingを利用
- 行動履歴をK個の興味カプセルにクラスタリング

**ComiRec-SA（Self-Attention版）**
- Transformerのマルチヘッドアテンション機構を利用
- 複数のアテンションヘッドが異なる興味を捉える
- DRより計算効率が高い

### 2. Aggregation Module（制御因子つき）

集約時に**制御因子 λ（0〜1）**を用いてスコアを計算：

```
Score(u, i) = λ * Relevance(u, i) + (1 - λ) * Diversity(u, i)
```

- λ = 1.0：純粋な関連性スコア（精度重視）
- λ = 0.0：純粋な多様性スコア
- λ の調整で精度・多様性のトレードオフを制御

### 3. 推薦時の動作（item側固定）

```
1. ユーザー → ComiRec → [u1, u2, ..., uK]（K個の興味ベクトル）
2. 各 ui でANNインデックス（固定）を検索 → 候補集合
3. 集約モジュール（λ制御）で最終スコアリング
```

## 本実験との関連性

### ✅ 適用可能な点

- **item側は固定のANNインデックスをそのまま利用**
- クエリ側のみで多様性を制御するという考え方が本実験と完全に一致
- **λパラメーターにより精度を保ちながら多様性を調整**できる
- ComiRec-SAはTransformerベースであり、言語情報埋め込みとの親和性が高い

### ✅ 言語情報埋め込みとの親和性

- 本実験のように言語モデルでアイテムを表現している場合、Self-Attentionベースの ComiRec-SA はユーザー興味の捉え方として相性が良い
- Transformerの多ヘッドアテンションが言語空間の複数の側面を捉えるメカニズムと類似

### ⚠️ 注意点

- 多様性指標と精度指標のトレードオフが λ 1つのパラメーターで制御されるため、柔軟性に限界がある
- 多様性の定義が「カバレッジ」ベースであり、推薦ごとの変化（temporal diversity）とは異なる

## 実験結果

| データセット | Hit Rate @50 | NDCG @50 | Coverage ↑ |
|---|---|---|---|
| Amazon | SOTA | SOTA | SOTA |
| Taobao | SOTA | SOTA | SOTA |

- MINDを含む全ての比較手法を精度・多様性ともに上回る

## 引用

```bibtex
@inproceedings{cen2020comirec,
  title={Controllable Multi-Interest Framework for Recommendation},
  author={Cen, Yukuo and Zhang, Jianwei and Zou, Xu and Zhou, Chang and Yang, Hongxia and Tang, Jie},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={2942--2951},
  year={2020}
}
```
