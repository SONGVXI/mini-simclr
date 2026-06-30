import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNNEncoder(nn.Module):
    """
    小型 CNN Encoder，用于 CIFAR-10 (32x32) 图像特征提取。

    参考 SimCLR 论文中的简化结构，输出 512 维特征向量。
    这个 encoder 用于 SimCLR 预训练，之后在 linear probe 阶段被冻结。
    """

    def __init__(self, feature_dim=512):
        """
        初始化小型 CNN Encoder。

        参数：
            feature_dim (int): 输出特征维度，默认 512
        """
        super().__init__()

        # 卷积层序列
        self.conv_layers = nn.Sequential(
            # 输入: 3x32x32 (CIFAR-10)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 64x16x16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 128x8x8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 256x4x4

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 输出: 512x1x1
        )

        # 全连接层将特征映射到指定维度
        self.fc = nn.Linear(512, feature_dim)

        print(f"小型 CNN Encoder 初始化完成，输出维度: {feature_dim}")

    def forward(self, x):
        """
        前向传播。

        参数：
            x (torch.Tensor): 输入图像张量，形状 [batch_size, 3, 32, 32]

        返回：
            torch.Tensor: 特征向量，形状 [batch_size, feature_dim]
        """
        # 卷积特征提取
        features = self.conv_layers(x)

        # 展平 [batch_size, 512, 1, 1] -> [batch_size, 512]
        features = features.view(features.size(0), -1)

        # 全连接层
        features = self.fc(features)

        return features


class SimCLRProjectionHead(nn.Module):
    """
    SimCLR Projection Head，将 encoder 特征映射到对比学习空间。

    参考论文结构: Linear -> ReLU -> Linear
    推荐维度: encoder_dim -> 512 -> 128
    输出需要 L2 归一化。
    """

    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        """
        初始化 Projection Head。

        参数：
            input_dim (int): 输入特征维度（encoder 输出维度）
            hidden_dim (int): 隐藏层维度，默认 512
            output_dim (int): 输出维度，默认 128
        """
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        print(f"Projection Head 初始化完成: {input_dim} -> {hidden_dim} -> {output_dim}")

    def forward(self, x):
        """
        前向传播。

        参数：
            x (torch.Tensor): encoder 输出特征，形状 [batch_size, input_dim]

        返回：
            torch.Tensor: L2 归一化后的投影向量，形状 [batch_size, output_dim]
        """
        # 线性投影
        z = self.projection_head(x)

        # L2 归一化（对比学习要求）
        z = F.normalize(z, dim=1)

        return z


class SimCLRModel(nn.Module):
    """
    SimCLR 完整模型，包含 encoder 和 projection head。

    结构:
    Image -> Encoder -> Projection Head -> L2 Normalize

    训练时使用完整结构，linear probe 时只使用 encoder。
    """

    def __init__(self, encoder=None, feature_dim=512, proj_hidden_dim=512, proj_output_dim=128):
        """
        初始化 SimCLR 模型。

        参数：
            encoder (nn.Module, optional): 自定义 encoder，默认使用 SmallCNNEncoder
            feature_dim (int): encoder 输出维度
            proj_hidden_dim (int): projection head 隐藏层维度
            proj_output_dim (int): projection head 输出维度
        """
        super().__init__()

        # 使用提供的 encoder 或创建默认 encoder
        if encoder is None:
            self.encoder = SmallCNNEncoder(feature_dim=feature_dim)
        else:
            self.encoder = encoder

        # projection head
        self.projection_head = SimCLRProjectionHead(
            input_dim=feature_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim
        )

        print(f"SimCLR 模型初始化完成:")
        print(f"  - Encoder 输出维度: {feature_dim}")
        print(f"  - Projection Head: {feature_dim} -> {proj_hidden_dim} -> {proj_output_dim}")

    def forward(self, x, return_features=False):
        """
        前向传播。

        参数：
            x (torch.Tensor): 输入图像，形状 [batch_size, 3, 32, 32]
            return_features (bool): 是否同时返回 encoder 特征

        返回：
            torch.Tensor: 投影向量（L2 归一化）
            如果 return_features=True: 返回 (投影向量, encoder特征)
        """
        # encoder 提取特征
        features = self.encoder(x)

        # projection head 映射到对比空间
        projections = self.projection_head(features)

        if return_features:
            return projections, features
        return projections

    def encode(self, x):
        """
        仅使用 encoder 提取特征（用于 linear probe 阶段）。

        参数：
            x (torch.Tensor): 输入图像，形状 [batch_size, 3, 32, 32]

        返回：
            torch.Tensor: encoder 特征，形状 [batch_size, feature_dim]
        """
        return self.encoder(x)


class LinearClassifier(nn.Module):
    """
    线性分类器，用于 SimCLR 线性评估阶段。

    在预训练完成后，冻结 encoder，训练这个线性分类器。
    """

    def __init__(self, input_dim=512, num_classes=10):
        """
        初始化线性分类器。

        参数：
            input_dim (int): 输入特征维度（encoder 输出维度）
            num_classes (int): 类别数量，CIFAR-10 为 10
        """
        super().__init__()

        self.classifier = nn.Linear(input_dim, num_classes)

        print(f"线性分类器初始化完成: {input_dim} -> {num_classes}")

    def forward(self, x):
        """
        前向传播。

        参数：
            x (torch.Tensor): encoder 特征，形状 [batch_size, input_dim]

        返回：
            torch.Tensor: 分类 logits，形状 [batch_size, num_classes]
        """
        return self.classifier(x)


def test_model():
    """测试模型类"""
    print("测试 SimCLR 模型...")

    # 创建 SimCLR 模型
    model = SimCLRModel()

    # 创建测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 32, 32)

    # 测试完整前向传播
    projections = model(test_input)
    print(f"投影输出形状: {projections.shape}")  # 应为 [4, 128]
    print(f"投影范数: {torch.norm(projections, dim=1).mean():.4f}")  # 应接近 1.0 (L2 归一化)

    # 测试同时返回特征
    projections2, features = model(test_input, return_features=True)
    print(f"Encoder 特征形状: {features.shape}")  # 应为 [4, 512]

    # 测试仅编码
    encoded_features = model.encode(test_input)
    print(f"仅编码特征形状: {encoded_features.shape}")  # 应为 [4, 512]

    # 测试线性分类器
    linear_classifier = LinearClassifier(input_dim=512, num_classes=10)
    logits = linear_classifier(features)
    print(f"分类器 logits 形状: {logits.shape}")  # 应为 [4, 10]

    # 检查模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    projection_params = sum(p.numel() for p in model.projection_head.parameters())

    print(f"\n模型参数统计:")
    print(f"  - Encoder 参数数量: {encoder_params:,}")
    print(f"  - Projection Head 参数数量: {projection_params:,}")
    print(f"  - 总参数数量: {total_params:,}")

    print("\n模型测试完成!")


if __name__ == "__main__":
    test_model()