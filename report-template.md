# Mini-SimCLR 图像表征学习复现实验报告

## 1. 论文信息

- 论文名称：A Simple Framework for Contrastive Learning of Visual Representations
- 论文地址：https://arxiv.org/abs/2002.05709
- 官方代码参考：https://github.com/google-research/simclr

## 2. 任务说明

本实验复现的任务是自监督图像表征学习。

```text
预训练输入：无标签图像
预训练目标：让同一图像的两种增强视图在表征空间中更接近，让不同图像的表征更远
评估方式：冻结 encoder，训练 linear probe，报告 CIFAR-10 分类准确率
```

## 3. 数据集

- 数据集名称：CIFAR-10
- 数据集地址：https://www.cs.toronto.edu/~kriz/cifar.html
- 实际使用预训练图像数：
- 实际使用 linear probe 训练图像数：
- 实际使用测试图像数：
- 使用设备：CPU / GPU（GPU）
- 总训练耗时：预训练约 5 分钟 + Linear Probe 约 2 分钟，总计约 7 分钟

## 4. 数据增强

请说明自己使用的增强策略：

| 增强方法 | 参数设置 |
|---|---|
| RandomResizedCrop | size=32, scale=(0.08, 1.0) |
| RandomHorizontalFlip | p=0.5 |
| ColorJitter | brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, 应用概率 p=0.8 |
| RandomGrayscale | p=0.2 |
| GaussianBlur | 未使用 |

请说明为什么这些增强适合 SimCLR：

```text
（在这里填写）
SimCLR 的核心思想是让同一张图像的不同增强视图在表征空间中尽可能接近，同时让不同图像的表征远离。因此，数据增强策略需要满足两个条件：
1. RandomResizedCrop（随机裁剪缩放）：迫使模型关注图像的局部特征而非全局布局，学习到尺度不变的表征。scale=(0.08, 1.0) 意味着裁剪范围从 8% 到 100%，提供了丰富的尺度变化。
2. RandomHorizontalFlip（水平翻转）：引入左右对称不变性，对于 CIFAR-10 中的大多数类别（如动物、交通工具）是合理的。
3. ColorJitter（颜色抖动）：通过改变亮度、对比度、饱和度和色调，让模型学习到颜色不变的表征，而不是依赖颜色捷径。SimCLR 论文表明颜色增强对性能提升至关重要。
4. RandomGrayscale（随机灰度化）：进一步迫使模型不依赖颜色信息，鼓励模型学习形状和纹理等更高层次的语义特征。
这些增强的组合保证了正样本对 (view1, view2) 共享高层语义但具有不同的低层视觉特征，这正是对比学习所需要的。
```

## 5. 模型结构

请说明自己的 Mini-SimCLR 结构：

```text
Image -> Two Augmented Views -> Shared Encoder -> Projection Head -> NT-Xent Loss
```

### 5.1 Encoder

- encoder 类型：小型 CNN 
- 输出特征维度：512
- 是否使用预训练权重：否

### 5.2 Projection Head

- MLP 层数：2
- hidden dimension：512
- output dimension：128
- 是否使用 ReLU / BatchNorm：使用 ReLU，不使用 BatchNorm

### 5.3 Linear Probe

- encoder 是否冻结：是
- linear classifier 输入维度：512
- 类别数：10

## 6. Loss 实现

请说明 NT-Xent loss 的实现方式：

- batch size：128
- `2N` 个增强样本如何构造：将 batch 内 N 张图像的两个增强视图 z_i 和 z_j 拼接为 [z_i, z_j]，得到 2N=256 个特征向量，形状为 [256, 128]
- 正样本索引如何确定：对于第 i 个样本（i < N），正样本是 i+N；对于第 i 个样本（i ≥ N），正样本是 i-N。使用 labels = cat([arange(N)+N, arange(N)]) 构造正样本标签
- temperature：0.5
- logits shape：[256, 256]

可以粘贴关键代码片段或伪代码。
```python
# 拼接特征: [z_i, z_j] -> [2N, D]
features = torch.cat([z_i, z_j], dim=0)
# 计算余弦相似度矩阵
similarity_matrix = F.cosine_similarity(
    features.unsqueeze(1),  # [2N, 1, D]
    features.unsqueeze(0),  # [1, 2N, D]
    dim=2
)
similarity_matrix = similarity_matrix / self.temperature
# 正样本标签: 样本 i 的正样本是 i+N 或 i-N
labels = torch.arange(batch_size, device=device)
labels = torch.cat([labels + batch_size, labels])
# 排除自身（对角线）
mask = torch.eye(2 * batch_size, dtype=torch.bool)
similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
# 交叉熵损失
loss = self.cross_entropy(similarity_matrix, labels)
```

## 7. 训练设置

### 7.1 自监督预训练

| 配置 | 数值 |
|---|---:|
| train images | 5000 |
| epochs | 10 |
| batch size | 128 |
| optimizer | Adam (weight_decay=1e-6) |
| learning rate | 1e-3（配合 CosineAnnealingLR 调度，eta_min=1e-6） |
| temperature | 0.5 |
| encoder | 小型 CNN（SmallCNNEncoder） |
| device | GPU (CUDA, NVIDIA RTX 4060) |

### 7.2 Linear Probe

| 配置 | 数值 |
|---|---:|
| train images | 5000 |
| test images | 2000 |
| epochs | 10 |
| batch size | 128 |
| optimizer | Adam |
| learning rate | 1e-3 |
| device | GPU (CUDA, NVIDIA RTX 4060) |

## 8. 训练过程

粘贴 contrastive loss 日志或 loss 曲线。

示例：

| Epoch | Contrastive Loss |
|---|---:|
| 1 | 5.4255 |
| 2 | 5.3078 |
| 3 | 5.2920 |
| 4 | 5.1883 |
| 5 | 5.1414 |
| 6 | 5.0864 |
| 7 | 5.0441 |
| 8 | 4.9882 |
| 9 | 4.9867 |
| 10 | 4.9481 |

请简要描述 loss 是否下降，以及训练是否稳定：

```text
（在这里填写）
Loss 从第 1 个 epoch 的 5.4255 稳定下降至第 10 个 epoch 的 4.9481，总下降约 0.4774。
训练过程非常稳定，没有出现震荡或发散现象。
Loss 下降速度在前期（epoch 1-4）较快，后期（epoch 7-10）趋于平缓，说明模型正在收敛。
配合 CosineAnnealingLR 学习率调度，学习率从 1e-3 逐渐衰减至约 1e-6，有助于训练后期的稳定收敛。
```

## 9. Linear Probe 结果

| 指标 | 结果 |
|---|---:|
| test accuracy |  |
| random baseline | 10% |

请分析结果是否明显高于随机猜测：

```text
（在这里填写）
```

## 10. 预测结果展示

至少展示 3 个测试样例。

| 图片编号 | 真实类别 | 预测类别 | 是否正确 |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |

## 11. 问题与改进

请简要说明：

- 遇到了哪些问题；
- 最终如何解决；
- 如果继续改进，可以从哪些方面入手，例如 batch size、epoch、temperature、projection head、数据增强等。

```text
（在这里填写）
```

## 12. AI 对话过程记录

- 录制工具：
- 对话链接：
- 使用的 AI 模型：
- 累计对话时长 / 会话数：

简要说明 AI 在哪些环节提供帮助，以及哪些部分是自己独立完成或验证的：

```text
（在这里填写）
```

## 13. Git 提交记录

- 仓库地址：
- 总 commit 数：

粘贴 `git log --oneline` 输出：

```text
（在这里粘贴 git log --oneline）
```
