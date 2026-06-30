import torch
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as transforms


class SimCLRDataset(Dataset):
    """
    SimCLR 数据集类，用于加载 CIFAR-10 并生成两种随机增强视图。

    根据 SimCLR 论文要求，至少实现以下增强中的 3 种：
    - RandomResizedCrop
    - RandomHorizontalFlip
    - ColorJitter
    - RandomGrayscale
    - GaussianBlur（可选）

    每张输入图像生成两次独立随机增强，得到 view_1 和 view_2。
    """

    def __init__(self, root='./data', train=True, download=True, num_samples=None,
                 image_size=32, s=1.0):
        """
        初始化 SimCLR 数据集。

        参数：
            root (str): 数据集存储路径
            train (bool): 是否使用训练集
            download (bool): 是否下载数据集
            num_samples (int, optional): 使用的图像数量，None 表示使用全部
            image_size (int): 输出图像尺寸
            s (float): 颜色增强强度系数
        """
        super().__init__()

        # 加载原始 CIFAR-10 数据集
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )

        # 如果指定了样本数量，则截取子集
        if num_samples is not None and num_samples < len(self.dataset):
            indices = torch.randperm(len(self.dataset))[:num_samples].tolist()
            self.dataset = Subset(self.dataset, indices)

        # 定义 SimCLR 数据增强管道
        # 参考论文：A Simple Framework for Contrastive Learning of Visual Representations
        # 增强包括：随机裁剪、水平翻转、颜色抖动、随机灰度化
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        # 数据增强管道（两个视图使用相同的增强集合但独立随机化）
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])

        print(f"SimCLR 数据集初始化完成，共 {len(self)} 张图像")
        print(f"增强管道：RandomResizedCrop -> RandomHorizontalFlip -> ColorJitter -> RandomGrayscale")

    def __len__(self):
        """返回数据集大小"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        获取第 idx 个样本，返回两个增强视图 (view1, view2)

        参数：
            idx (int): 样本索引

        返回：
            tuple: (view1, view2) 两个增强后的图像张量
        """
        # 获取原始图像
        image, _ = self.dataset[idx]  # 忽略标签，SimCLR 是无监督学习

        # 应用两次独立的随机增强
        # 由于 transform 中包含随机操作，每次调用都会产生不同的增强
        view1 = self.transform(image)
        view2 = self.transform(image)

        return view1, view2


class LinearProbeDataset(Dataset):
    """
    线性评估数据集类，用于训练线性分类器。

    在预训练完成后，冻结 encoder，使用标签训练线性分类器。
    这个数据集返回图像和对应的标签。
    """

    def __init__(self, root='./data', train=True, download=True, num_samples=None):
        """
        初始化线性评估数据集。

        参数：
            root (str): 数据集存储路径
            train (bool): 是否使用训练集
            download (bool): 是否下载数据集
            num_samples (int, optional): 使用的图像数量，None 表示使用全部
        """
        super().__init__()

        # 定义标准数据预处理（与 SimCLR 评估时一致）
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])

        # 加载 CIFAR-10 数据集
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=self.transform
        )

        # 如果指定了样本数量，则截取子集
        if num_samples is not None and num_samples < len(self.dataset):
            indices = torch.randperm(len(self.dataset))[:num_samples].tolist()
            self.dataset = Subset(self.dataset, indices)

        print(f"线性评估数据集初始化完成，共 {len(self)} 张图像")
        print(f"使用标签：{'训练集' if train else '测试集'}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        获取第 idx 个样本

        参数：
            idx (int): 样本索引

        返回：
            tuple: (image, label) 图像张量和标签
        """
        return self.dataset[idx]


def test_dataset():
    """测试数据集类"""
    print("测试 SimCLR 数据集...")

    # 创建 SimCLR 数据集（使用 100 张图像进行测试）
    simclr_dataset = SimCLRDataset(
        root='./data',
        train=True,
        download=True,
        num_samples=5000,
        image_size=32
    )

    # 检查数据集大小
    print(f"数据集大小: {len(simclr_dataset)}")

    # 获取一个样本
    view1, view2 = simclr_dataset[0]
    print(f"View 1 形状: {view1.shape}")
    print(f"View 2 形状: {view2.shape}")
    print(f"View 1 数据类型: {view1.dtype}")
    print(f"View 1 值范围: [{view1.min():.3f}, {view1.max():.3f}]")

    # 检查两个视图是否不同（由于随机增强）
    diff = torch.abs(view1 - view2).mean().item()
    print(f"两个视图的平均绝对差异: {diff:.4f}")
    print(f"视图是否不同: {diff > 0.001}")

    print("\n测试线性评估数据集...")

    # 创建线性评估数据集
    linear_dataset = LinearProbeDataset(
        root='./data',
        train=False,  # 使用测试集
        download=True,
        num_samples=50
    )

    # 获取一个样本
    image, label = linear_dataset[0]
    print(f"图像形状: {image.shape}")
    print(f"标签: {label}")
    print(f"标签类型: {type(label)}")

    print("\n数据集测试完成！")


if __name__ == "__main__":
    test_dataset()