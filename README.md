# Mini-SimCLR 图像表征学习复现作业

## 1. 复现论文

本次复现参考论文：

**A Simple Framework for Contrastive Learning of Visual Representations**

- 论文作者：Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
- 论文地址：https://arxiv.org/abs/2002.05709
- PDF：https://arxiv.org/pdf/2002.05709
- 官方代码参考：https://github.com/google-research/simclr

本次作业不要求完整复现论文中的 ImageNet 大规模实验，只要求完成一个轻量化版本的 SimCLR，并在 CIFAR-10 小子集上跑通自监督预训练和线性评估流程。

考虑到同学需要在一周内使用个人电脑完成，本作业按 **CPU 友好版本** 设计。评价重点是复现思路、模块实现和训练流程是否正确，不以高准确率作为硬性要求。

## 2. 复现目标

实现一个简化版 SimCLR，用于学习图像表征。

任务形式：

```text
输入：一批无标签图像
输出：图像表征向量
评估：冻结表征编码器，训练线性分类器，在 CIFAR-10 测试集上报告分类准确率
```

核心目标：

1. 对同一张图像生成两种随机增强视图；
2. 使用 CNN/ResNet 编码器提取图像特征；
3. 使用 projection head 将特征映射到对比学习空间；
4. 实现 NT-Xent contrastive loss；
5. 完成自监督预训练；
6. 冻结 encoder，训练 linear probe，并报告测试准确率。

## 3. 数据集要求

使用公开数据集：**CIFAR-10**

- 官方数据集地址：https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch 可通过 `torchvision.datasets.CIFAR10` 自动下载。

数据规模：

- 最低要求：使用 CIFAR-10 train split 中至少 1000 张图像进行自监督预训练；
- 推荐要求：使用 2000-5000 张训练图像进行自监督预训练；
- 挑战要求：使用完整 50000 张训练图像进行自监督预训练；
- 评估要求：使用 CIFAR-10 test split 中至少 1000 张图像进行 linear probe 测试。

注意：

- 自监督预训练阶段不能使用标签；
- 标签只能用于 linear probe 训练和最终评估；
- 数据集不要提交到 Git 仓库。

## 4. 模型结构要求

需要按照 SimCLR 的核心思想实现如下结构：

```text
Image
  -> Random Augmentation View 1
  -> Random Augmentation View 2
  -> Shared Encoder
  -> Projection Head
  -> NT-Xent Contrastive Loss
```

推荐结构：

```text
Augmented Image
  -> ResNet-18 或小型 CNN Encoder
  -> Global Average Pooling
  -> MLP Projection Head
  -> L2 Normalize
  -> Contrastive Loss
```

### 4.1 数据增强

至少实现以下增强中的 3 种：

- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `ColorJitter`
- `RandomGrayscale`
- `GaussianBlur`（可选）

每张输入图像需要生成两次独立随机增强，得到 `view_1` 和 `view_2`。

### 4.2 Encoder

可选实现：

- 最低要求：自行实现一个小型 CNN encoder；
- 推荐要求：简化 ResNet-18，适配 CIFAR-10 的 32x32 输入。

要求：

- `view_1` 和 `view_2` 必须共享同一个 encoder；
- encoder 输出一个图像表征向量；
- linear probe 阶段需要能够冻结 encoder。

### 4.3 Projection Head

实现一个 MLP projection head，例如：

```text
Linear -> ReLU -> Linear
```

推荐维度：

```text
encoder_dim -> 512 -> 128
```

### 4.4 NT-Xent Loss

需要自行实现 NT-Xent loss，不能只调用现成库的一行封装。

最低要求：

1. 对 batch 内 `2N` 个增强样本计算 pairwise cosine similarity；
2. 正样本为同一张原图的两种增强视图；
3. 其他 `2N - 2` 个样本为负样本；
4. 使用 temperature 参数，推荐 `temperature = 0.5`；
5. 使用 cross entropy 形式计算 loss。

## 5. 训练要求

### 5.1 自监督预训练

基本设置建议：

```text
dataset: CIFAR-10 train split
train images: at least 1000
epochs: at least 3
batch size: 32 or 64
optimizer: Adam or SGD
learning rate: 1e-3 for Adam, or 0.03 for SGD
loss: NT-Xent
temperature: 0.5
```

需要记录每个 epoch 的 contrastive loss。

个人电脑建议配置：

```text
CPU 版本：1000-2000 张图像，小型 CNN，pretrain 3-5 epoch，linear probe 3 epoch
入门 GPU 版本：5000 张图像，小型 CNN 或 ResNet-18，pretrain 5-10 epoch，linear probe 5 epoch
挑战版本：完整 CIFAR-10，ResNet-18，更长 epoch
```

### 5.2 Linear Probe

预训练结束后：

1. 丢弃 projection head；
2. 冻结 encoder；
3. 在 encoder 后接一个线性分类器；
4. 使用 CIFAR-10 标签训练分类器；
5. 在测试集上报告 accuracy。

基本设置建议：

```text
epochs: at least 3
optimizer: Adam or SGD
loss: cross entropy
metric: test accuracy
```

最低结果要求：

- 必须跑通完整训练和评估流程；
- linear probe accuracy 不作为硬性门槛；
- 如果准确率接近随机猜测的 10%，需要在报告中分析原因，例如训练 epoch 太少、batch size 太小、增强过强、encoder 太浅等。

## 6. 最低完成标准

只要完成以下内容，即认为达到最低复现要求：

1. 能够加载 CIFAR-10，并为同一张图片生成两种随机增强视图；
2. 能够搭建 SimCLR 模型结构：encoder + projection head；
3. 能够正确实现 NT-Xent loss；
4. 能够完成自监督预训练流程，并保存 loss 记录；
5. 能够冻结 encoder，完成 linear probe 训练；
6. 能够报告测试 accuracy，并展示至少 5 张测试图像的预测类别。

## 7. 最终提交内容

最终提交内容包括：

1. **实现代码**

   需要包含数据读取、数据增强、模型定义、loss 实现、预训练脚本、linear probe 脚本、测试脚本。

2. **实验报告**

   说明复现论文、使用数据集、模型结构、数据增强策略、loss 实现、训练设置、实验结果和问题分析。

3. **训练过程记录**

   提供 contrastive loss 曲线或关键日志截图，说明模型确实完成了训练流程。

4. **Linear probe 结果**

   至少报告：

   ```text
   使用训练图像数：
   预训练 epoch：
   linear probe epoch：
   test accuracy：
   ```

5. **预测结果展示**

   展示至少 5 张 CIFAR-10 测试图片，给出真实类别和模型预测类别。
   
7. **学生仓库结构**
```text
student-mini-simclr/
├── README.md:说明项目环境、安装依赖、如何训练、如何评估；
├── requirements.txt:列出 Python 依赖
├── .gitignore
├── code/存放全部实现代码
├── report/实验报告,loss 曲线、预测样例图等报告图片
├── logs/训练过程日志
└── results/保存最终 accuracy、训练数据量、epoch、batch size 等结果
```

## 8. 评分建议

总分 100 分：

| 模块 | 分值 | 要求 |
|---|---:|---|
| 数据读取与增强 | 15 | 正确加载 CIFAR-10，能生成两种增强视图 |
| 模型结构 | 20 | 实现 encoder、projection head，并能输出表征 |
| NT-Xent loss | 20 | 正确构造正负样本与 temperature-scaled logits |
| 训练流程 | 20 | 跑通自监督预训练和 linear probe |
| 实验报告 | 15 | 结构完整，包含 loss、accuracy、问题分析 |
| 过程记录 | 10 | 提交 AI 对话记录、Git 小步提交记录 |

加分项，最多 10 分：

- 对比不同 temperature 或 batch size 的效果；
- 对比有无 projection head 的效果；
- 可视化 learned embedding，例如 t-SNE 或 UMAP；
- 展示 nearest neighbor retrieval 结果。

## 9. 过程记录与防作业要求

为了确认作业是本人逐步完成，而不是一次性生成成品代码，本次复现必须满足以下两条过程性要求。

### 9.1 AI 对话全过程记录

要求使用 entir.io 或同类可分享的对话记录工具，记录与 AI 工具的全部开发对话，并在提交时附上可访问链接。

要求：

```text
- 覆盖范围：从读数据、写 augmentation、实现 loss、训练、调 bug、写报告的全过程
- 不能只录最后一段成品对话，中间的试错、报错、追问都要保留
- 链接需要可访问，公开或对老师/助教开放
```

提交示例：

```text
AI 对话记录：https://entir.io/s/xxxxxx
使用模型：ChatGPT / Claude / Gemini
累计对话时长：约 3 小时，分 5 次会话
```

### 9.2 Git 小步提交

要求每完成一个小模块提交一次 commit，禁止一次性把整个项目 push 上去。

合格 commit 粒度示例：

```text
feat: 加载 CIFAR-10 并实现双视图数据增强
feat: 实现 ResNet/CNN encoder
feat: 添加 projection head
feat: 实现 NT-Xent loss
feat: 完成 SimCLR pretrain loop
feat: 添加 linear probe 训练流程
fix: 修复 contrastive logits 中正样本索引错误
feat: 添加测试集预测展示脚本
docs: 补充实验报告中的 loss 曲线和 accuracy 表格
```

提交时一并附上：

```text
仓库地址：GitHub / Gitee 均可
git log --oneline 的文本输出或截图
```
