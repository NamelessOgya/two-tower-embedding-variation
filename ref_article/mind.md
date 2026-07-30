# MIND: Multi-Interest Network with Dynamic Routing for Recommendation at Tmall

## 基本情報

| 項目 | 内容 |
|---|---|
| タイトル | Multi-Interest Network with Dynamic Routing for Recommendation at Tmall |
| 著者 | Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Pipei Huang, Huan Zhao, Guoliang Kang, Qiwei Chen, Wei Li, Dik Lun Lee |
| 所属 | Alibaba Group |
| 会議 | CIKM 2019 |
| arxiv | [1904.08030](https://arxiv.org/abs/1904.08030) |
| 発表年 | 2019 |

## 概要

Tmallの大規模推薦システムにおいて、ユーザーを**単一ベクトルではなく複数ベクトル**で表現する手法を提案した先駆的研究。
カプセルネットワークのルーティング機構を応用した「多興味抽出層（Multi-Interest Extractor Layer）」を設計し、ユーザーの多様な興味を複数の興味ベクトルとして獲得する。

## 問題設定

- 既存のDeep Learning推薦モデルはユーザーを**1つのベクトル**で表現する
- しかし現実のユーザーは**複数の異なる興味**を持っており、単一ベクトルではそれを表現しきれない
- 結果として、興味の一側面に偏った推薦になりがち

## 提案手法：MIND

### 1. Multi-Interest Extractor Layer（多興味抽出層）

- カプセルネットワーク（Capsule Network）のDynamic Routingを応用
- ユーザーの行動履歴アイテムを複数の「興味カプセル（Interest Capsules）」にクラスタリング
- 各カプセルがユーザーの異なる興味側面を表現するベクトルになる

### 2. Label-Aware Attention（ラベル認識アテンション）

- 訓練時に「ターゲットアイテム」に最も関連する興味ベクトルを選択するためのアテンション機構
- ユーザー表現の品質を向上させるために導入

### 3. 推薦時の動作

```
1. ユーザーの行動履歴 → Multi-Interest Extractor → K個の興味ベクトル [u1, u2, ..., uK]
2. 各興味ベクトルでANN検索 → K個の候補集合の結合
3. 全候補から最終的なTop-N候補を選択
```

- **item側のベクトルは変更しない**
- クエリ側（ユーザー）が複数ベクトルを持つことで多様性を実現

## 本実験との関連性

### ✅ 適用可能な点

- **被検索側（item）のベクトルを変えない**：item embeddingは固定されたANNインデックスをそのまま利用
- **クエリ側のみ変化**：ユーザーの複数の興味ベクトルを推薦ごとに使い分けることで、多様な候補を取得
- 毎回異なる興味ベクトルを選択する確率的な選択戦略と組み合わせれば、**推薦するたびに候補を変える**多様性が実現できる

### ⚠️ 注意点

- カプセルルーティングはパラメーターチューニングが複雑
- 興味ベクトルの数Kは事前に設定する必要がある
- 論文の主目的は精度向上であり、多様性は副次的効果

## 実験結果

| データセット | 手法 | Hit Rate @50 | NDCG @50 |
|---|---|---|---|
| Amazon Books | MIND | **最高** | **最高** |
| Tmall | MIND | **最高** | **最高** |

- 既存のYoutubeDNNやGRU4Rec等を上回る性能を達成
- 本番Tmall Mobile Appにデプロイ済み

## 引用

```bibtex
@inproceedings{li2019mind,
  title={Multi-Interest Network with Dynamic Routing for Recommendation at Tmall},
  author={Li, Chao and Liu, Zhiyuan and Wu, Mengmeng and Xu, Yuchi and Huang, Pipei and Zhao, Huan and Kang, Guoliang and Chen, Qiwei and Li, Wei and Lee, Dik Lun},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM)},
  pages={2615--2623},
  year={2019}
}
```
