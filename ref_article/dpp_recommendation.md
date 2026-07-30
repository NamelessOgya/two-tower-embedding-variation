# Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity

## 基本情報

| 項目 | 内容 |
|---|---|
| タイトル | Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity |
| 著者 | Laming Chen, Guoxin Zhang, Eric Zhou |
| 所属 | Tencent AI Lab |
| 会議 | NeurIPS 2018 |
| arxiv | [1709.05135](https://arxiv.org/abs/1709.05135) |
| 発表年 | 2018 |

## 概要

**行列式点過程（Determinantal Point Process, DPP）**を用いた推薦多様化手法を、大規模な実用システムに適用可能にする高速アルゴリズムを提案。
DPPは関連性と多様性を確率的にモデル化できる数学的に優れた枠組みだが、MAPインフェランスがNP困難という問題があった。本論文では大幅に高速な貪欲アルゴリズムを提案した。

## 背景：DPPとは

### 数学的定義

アイテム全体の集合 $Y$ から部分集合 $S$ を選択する確率を次のように定義：

$$\mathcal{P}(S) \propto \det(\mathbf{L}_S)$$

ここで $\mathbf{L}_S$ は アイテム集合 $S$ のカーネル行列（$L$行列）のサブ行列。

### L行列の構成

$$L_{ij} = q_i \cdot \phi_i^T \phi_j \cdot q_j$$

| 記号 | 意味 |
|---|---|
| $q_i$ | アイテム $i$ の関連性スコア（クエリとの内積等） |
| $\phi_i$ | アイテム $i$ の正規化特徴ベクトル（多様性のため） |
| $\phi_i^T \phi_j$ | アイテム間の類似度（類似が高いほど行列式が小さくなる） |

### 行列式の意味

- $\det(\mathbf{L}_S)$ が大きい ← アイテムが互いに異なり（多様）、かつ各アイテムが関連性が高い
- 行列式は「平行四辺形の体積」に相当し、選択集合の多様性を幾何学的に表現

## MAPインフェランスの問題と解決

### 問題

$$S^* = \arg\max_S \det(\mathbf{L}_S)$$

この最適化問題はNP困難（組み合わせ爆発）

### 提案手法：高速貪欲アルゴリズム

```python
S = {}  # 空集合からスタート
candidates = all_items

while len(S) < N:
    # 各候補について行列式の増分を計算
    best_item = argmax over i in candidates:
        det(L_{S + {i}}) / det(L_S)  # 行列式比（= 条件付き行列式）
    
    S.add(best_item)
    candidates.remove(best_item)
```

**高速化のポイント：**
- コレスキー分解（Cholesky decomposition）を用いて行列式の逐次更新を効率化
- 従来のNP困難から**多項式時間**に削減
- 大規模推薦システムに実用的に適用可能

## 本実験との関連性

### ✅ 適用可能な点

- **item側のベクトルは変更しない**：DPPは候補集合のポスト処理として機能
- Two-Tower ModelのANN検索後に適用することで、多様性を担保できる
- 関連性（$q_i$）と多様性（$\phi_i^T \phi_j$）を明示的に分離してモデル化するため、精度を保ちながら多様性を向上できる

### ✅ Two-Tower Modelとの組み合わせ方

```
1. Two-Tower Model でユーザーベクトル生成
2. ANNインデックス（item側固定）で Top-M 候補取得（M >> N）
3. 各候補の関連性スコア q_i = dot(user_vec, item_vec)
4. 各候補の特徴ベクトル φ_i = item_embedding（正規化済み）
5. DPP MAP推論で多様なN件を選択
6. 最終推薦リスト（N件）を返す
```

### ✅ MMRとの比較

| 項目 | MMR | DPP |
|---|---|---|
| 数学的根拠 | ヒューリスティック | 確率論的に正当（行列式） |
| 多様性の表現 | ペアワイズ最大類似度 | 集合全体のボリューム |
| 計算量 | O(N²) | O(N² M)（高速化版） |
| チューニング | λ（1パラメーター） | L行列全体 |
| 実装の容易さ | 簡単 | やや複雑 |

### ⚠️ 注意点

- DPPのL行列の設計（特に多様性特徴 φ_i の定義）がパフォーマンスに大きく影響
- 言語埋め込みのitem vectorをそのまま φ_i として使えるが、適切な正規化が必要
- 候補数Mが大きいと計算コストが増大する

## 実験結果

- Tencent News推薦システムでの実験：**精度（CTR）を維持しながら多様性指標を改善**
- オフライン実験とオンラインA/Bテスト両方で有効性を確認
- 従来のDPP手法と比較して計算速度を大幅に改善（実用化に成功）

## 推薦多様性の指標（本実験でも参考になる指標）

| 指標 | 定義 | 用途 |
|---|---|---|
| Intra-List Diversity (ILD) | 推薦リスト内アイテム間の平均距離 | リスト内多様性 |
| Coverage | 全アイテムカテゴリのうち推薦でカバーされた割合 | カタログカバレッジ |
| Serendipity | ユーザーが予期しなかった良い推薦の割合 | 意外性 |
| Temporal Diversity | 推薦実行ごとに異なるアイテムが出る割合 | 時間的多様性（本実験の目標） |

## 引用

```bibtex
@inproceedings{chen2018fast,
  title={Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity},
  author={Chen, Laming and Zhang, Guoxin and Zhou, Eric},
  booktitle={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}
```
