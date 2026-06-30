import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss 实现。

    用于 SimCLR 对比学习，计算同一张图像两个增强视图之间的对比损失。
    参考论文: A Simple Framework for Contrastive Learning of Visual Representations

    损失计算步骤:
    1. 对 batch 内 2N 个增强样本计算 pairwise cosine similarity
    2. 正样本为同一张原图的两种增强视图 (i 和 i+N)
    3. 其他 2N-2 个样本为负样本
    4. 使用 temperature 参数缩放相似度
    5. 使用 cross entropy 形式计算 loss
    """

    def __init__(self, temperature=0.5, eps=1e-8):
        """
        初始化 NT-Xent Loss。

        参数：
            temperature (float): 温度参数，用于缩放相似度，默认 0.5
            eps (float): 数值稳定性的小常数，默认 1e-8
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        print(f"NT-Xent Loss 初始化完成，temperature = {temperature}")

    def forward(self, z_i, z_j):
        """
        计算 NT-Xent 损失。

        参数：
            z_i (torch.Tensor): 第一个增强视图的特征，形状 [batch_size, feature_dim]
            z_j (torch.Tensor): 第二个增强视图的特征，形状 [batch_size, feature_dim]
                                假设 z_i[k] 和 z_j[k] 是同一张图像的两个增强

        返回：
            torch.Tensor: 标量损失值
        """
        batch_size = z_i.size(0)

        # 验证输入形状
        assert z_i.shape == z_j.shape, "z_i 和 z_j 形状必须相同"
        assert len(z_i.shape) == 2, "输入必须是二维特征向量"

        # 拼接所有特征: [z_i, z_j] -> 形状 [2*batch_size, feature_dim]
        features = torch.cat([z_i, z_j], dim=0)

        # 计算余弦相似度矩阵
        # features @ features.T -> 形状 [2*batch_size, 2*batch_size]
        similarity_matrix = F.cosine_similarity(
            features.unsqueeze(1),  # [2N, 1, D]
            features.unsqueeze(0),  # [1, 2N, D]
            dim=2
        )

        # 除以温度参数
        similarity_matrix = similarity_matrix / self.temperature

        # 创建正样本掩码
        # 对于样本 i，正样本是 i+N (如果 i < N) 或 i-N (如果 i >= N)
        device = features.device
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels])  # [0,1,2,...,2N-1] 对应关系

        # 创建对角线掩码（排除自身）
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

        # 计算交叉熵损失
        loss = self.cross_entropy(similarity_matrix, labels)

        return loss

    def compute_similarity_matrix(self, features):
        """
        计算特征之间的余弦相似度矩阵（用于调试和可视化）。

        参数：
            features (torch.Tensor): 特征矩阵，形状 [N, D]

        返回：
            torch.Tensor: 余弦相似度矩阵，形状 [N, N]
        """
        # 标准化特征
        features_norm = F.normalize(features, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.mm(features_norm, features_norm.t())

        return similarity_matrix

    def get_positive_pairs(self, batch_size):
        """
        获取正样本对索引（用于调试）。

        参数：
            batch_size (int): 原始 batch 大小

        返回：
            list: 正样本对列表 [(i, j), ...]
        """
        positive_pairs = []
        for i in range(batch_size):
            positive_pairs.append((i, i + batch_size))
            positive_pairs.append((i + batch_size, i))
        return positive_pairs


class ContrastiveLoss(nn.Module):
    """
    另一种形式的对比损失，更直观的实现。

    与 NTXentLoss 功能相同，但实现方式更直接，便于理解。
    """

    def __init__(self, temperature=0.5):
        """
        初始化对比损失。

        参数：
            temperature (float): 温度参数，默认 0.5
        """
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        """
        计算对比损失（InfoNCE 形式）。

        参数：
            z_i (torch.Tensor): 第一个增强视图的特征
            z_j (torch.Tensor): 第二个增强视图的特征

        返回：
            torch.Tensor: 标量损失值
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # 拼接所有特征
        features = torch.cat([z_i, z_j], dim=0)  # [2N, D]

        # 计算相似度矩阵
        sim_matrix = torch.mm(features, features.t()) / self.temperature  # [2N, 2N]

        # 创建正样本标签
        # 样本 i 的正样本是 i+N，样本 i+N 的正样本是 i
        labels = torch.cat([
            torch.arange(batch_size, batch_size * 2, device=device),
            torch.arange(0, batch_size, device=device)
        ])  # [N, N+1, N+2, ..., 2N-1, 0, 1, 2, ..., N-1]

        # 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


def test_loss():
    """测试损失函数"""
    print("测试 NT-Xent Loss...")

    # 创建损失函数
    ntxent_loss = NTXentLoss(temperature=0.5)

    # 创建测试数据
    batch_size = 4
    feature_dim = 128

    # 模拟 SimCLR 投影输出
    z_i = torch.randn(batch_size, feature_dim)  # 第一个增强视图
    z_j = torch.randn(batch_size, feature_dim)  # 第二个增强视图

    # 计算损失
    loss = ntxent_loss(z_i, z_j)
    print(f"NT-Xent Loss 值: {loss.item():.4f}")

    # 测试相似度矩阵计算
    print("\n测试相似度矩阵计算...")
    features = torch.cat([z_i, z_j], dim=0)
    sim_matrix = ntxent_loss.compute_similarity_matrix(features)
    print(f"相似度矩阵形状: {sim_matrix.shape}")  # 应为 [8, 8]
    print(f"相似度矩阵范围: [{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]")

    # 测试正样本对
    positive_pairs = ntxent_loss.get_positive_pairs(batch_size)
    print(f"\n正样本对数量: {len(positive_pairs)}")
    print(f"示例正样本对: {positive_pairs[:4]}")

    # 测试 ContrastiveLoss
    print("\n测试 ContrastiveLoss...")
    contrastive_loss = ContrastiveLoss(temperature=0.5)
    loss2 = contrastive_loss(z_i, z_j)
    print(f"ContrastiveLoss 值: {loss2.item():.4f}")

    # 验证两个损失函数结果相近
    print(f"\n两个损失函数差异: {abs(loss.item() - loss2.item()):.6f}")

    print("\n损失函数测试完成!")


if __name__ == "__main__":
    test_loss()