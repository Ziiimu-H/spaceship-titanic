# Spaceship Titanic

一个完整的 Kaggle 入门项目：从随机森林开始，加入特征工程和 CatBoost，记录没有带来稳定提升的调参实验，最后复现公开 TabPFN 预测的融合方案。

**最佳 Kaggle 公开分数：0.82604。自行训练方案的最佳分数：0.80593。** 最佳成绩复用了 TaeyanG4 公开的 TabPFN 概率，并在本地重新训练元模型，没有自行训练 TabPFN。

## 结果

以下分数均来自本项目账号的实际提交；第四轮两份成绩已通过提交截图确认。

| 阶段 | 方法 | Kaggle 公开分数 | 结果说明 |
|---|---|---:|---|
| 第一版 | 随机森林 | 0.80032 | 自行训练 |
| 第二版 | CatBoost + 特征工程 | 0.80593 | 自行训练方案的最佳成绩 |
| 第三轮 | 特征消融与有限调参 | 0.80593 | 候选未通过另一套划分复查，沿用第二版预测 |
| 第四轮 A | 四路公开 TabPFN 概率 + 逻辑回归 | **0.82604** | 精确复现公开预测方案 |
| 第四轮 B | TabPFN + CatBoost 融合 | 0.82534 | 与 A 仅有 7 条测试预测不同 |

完整过程见 [项目总结](docs/PROJECT_SUMMARY.md)，机器可读成绩见 [reports/public_scores.csv](reports/public_scores.csv)。

## 快速开始

需要 Python 3.11。建议使用独立环境：

```bash
git clone https://github.com/Ziiimu-H/spaceship-titanic.git
cd spaceship-titanic
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

从 [比赛 Data 页面](https://www.kaggle.com/competitions/spaceship-titanic/data)下载 `train.csv`、`test.csv`、`sample_submission.csv`，放入 `data/`。数据和运行缓存不提交到本仓库。

### 自行训练

```bash
# 随机森林基线
python -m spaceship.train --model random-forest --features base --output-dir runs/random-forest

# CatBoost + 完整特征
python -m spaceship.train --model catboost --features full --output-dir runs/catboost
```

默认采用按同行组划分的五折验证。输出包含 `submission.csv`、折外概率、每折分数和参数记录。

需要复查历史随机分层结果时显式指定 `--split stratified`。特征消融可使用 `--features no_surname`、`no_group_id`、`no_group_fill` 或 `no_spend_features`；参数可通过 `--iterations`、`--depth`、`--l2` 调整。

### 复现公开 TabPFN 预测方案

```bash
# 下载并校验公开概率，不执行远程 Notebook 源码
python -m spaceship.assets

# 重新拟合本地元模型，生成两份提交文件
python -m spaceship.replay

# 核对乘客顺序、概率、标签以及来源文件校验值
python -m spaceship.verify
```

提交文件位于：

- `runs/replay/submission_tabpfn_baseline.csv`
- `runs/replay/submission_tabpfn_catboost_blend.csv`

两份文件各有 4,277 行，与来源给出的提交文件 SHA-256 完全一致。这个过程在 CPU 上即可运行，不下载 TabPFN 权重。

### Notebook 与测试

```bash
python -m jupyterlab notebooks
python -m unittest discover -s tests -v
```

- [01：数据、特征与自行训练](notebooks/01_training.ipynb)
- [02：公开预测复现与结果检查](notebooks/02_public_prediction_replay.ipynb)

## 代码结构

```text
spaceship/
  features.py       # 缺失值处理、舱位与消费特征、无标签名册上下文
  ablation.py       # 独立移除各组特征
  train.py          # 随机森林 / CatBoost 训练与交叉验证
  assets.py         # 下载并校验公开概率包
  replay.py         # 逻辑回归、样条元模型与概率融合
  verify.py         # 独立复核导出结果
notebooks/          # 可运行说明
reports/            # 历史实验表与实际提交分数
reference/          # 来源、许可说明与固定校验值
```

## 验证与边界

本数据的训练集和测试集没有共享同行组。项目早期使用随机分层划分，后来识别出这种划分允许同组乘客跨训练折与验证折，因此发布版默认改为分组验证。两类验证分数不能直接比较。

增强特征使用全部公开乘客资料统计同行人数、同姓人数等，不使用测试答案。这是面向已知乘客名册的上下文特征，不能据此断言对全新名册同样有效。

公开 TabPFN 方案使用固定的基础模型折外预测，再验证元模型；它不是完整嵌套重训。其开发分数不能视为独立最终评估。公开榜成绩也不能替代独立测试。

## 来源与署名

- [比赛：Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
- [Gusthema：TFDF 入门 Notebook](https://www.kaggle.com/code/gusthema/spaceship-titanic-with-tfdf)
- [werenja：CatBoost 与特征工程](https://www.kaggle.com/code/werenja/spaceship-titanic-top-4-catboost-feature-eng)
- [TaeyanG4：公开 TabPFN 概率](https://www.kaggle.com/datasets/taeyangg4/spaceship-titanic-tabpfn-member-predictions)
- [kalidenaiarys：TabPFN + CatBoost 复现](https://www.kaggle.com/code/kalidenaiarys/spaceship-titanic-tabpfn-catboost-lb-0-82534)

公开 TabPFN 概率依来源按 **CC BY 4.0** 保留署名；下载至本地缓存后保持原文件不变。见 [第三方说明](reference/ATTRIBUTION.md)。模型权重未包含在仓库中。项目的代码整理与实验在 AI 协助下完成。
