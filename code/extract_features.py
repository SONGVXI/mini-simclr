"""
SimCLR 图像表征提取演示脚本。

演示核心流程：输入一批无标签图像 → 输出图像表征向量。
这也是 SimCLR 预训练后 encoder 的实际用途——将任意图像映射为固定维度的表征向量。

用法：
    python extract_features.py
"""

import torch
import numpy as np
import os
import sys

# 添加 code 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import SimCLRDataset
from model import SmallCNNEncoder


def extract_features(encoder, images, device):
    """
    核心函数：输入一批无标签图像，输出图像表征向量。

    这就是 SimCLR 的核心能力——学到一个函数 f: Image → Vector，
    使得语义相似的图像在向量空间中距离更近。

    参数：
        encoder (nn.Module): 预训练好的 encoder
        images (torch.Tensor): 一批图像，形状 [batch_size, 3, 32, 32]
        device (torch.device): 计算设备

    返回：
        torch.Tensor: 图像表征向量，形状 [batch_size, feature_dim]
    """
    encoder.eval()
    with torch.no_grad():
        images = images.to(device)
        features = encoder(images)  # 输入图像 → 输出表征向量
    return features


def demonstrate_pipeline():
    """
    完整演示：输入无标签图像 → 输出表征向量。
    """
    # ============================================================
    # 配置
    # ============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder_path = './results/simclr_encoder.pth'
    data_dir = './data'
    feature_dim = 512
    num_demo_images = 8  # 演示用的图像数量

    print("=" * 60)
    print("SimCLR 图像表征提取演示")
    print("=" * 60)
    print(f"设备: {device}")
    print()

    # ============================================================
    # Step 1: 加载预训练 Encoder
    # ============================================================
    print("Step 1: 加载预训练 Encoder")
    print("-" * 40)

    encoder = SmallCNNEncoder(feature_dim=feature_dim).to(device)

    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"  ✓ 成功加载预训练权重: {encoder_path}")
    else:
        print(f"  ⚠ 未找到预训练权重 {encoder_path}")
        print(f"  使用随机初始化的 encoder（表征无意义）")
        print(f"  请先运行 python pretrain.py 进行预训练")

    encoder.eval()
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder 参数量: {total_params:,}")
    print()

    # ============================================================
    # Step 2: 加载一批无标签图像
    # ============================================================
    print("Step 2: 加载一批无标签图像（无监督，不使用标签）")
    print("-" * 40)

    # 使用 SimCLRDataset 加载图像（返回两个增强视图，但我们只需要原图）
    # 注意：这里我们只演示 encoder 的推理，所以用 LinearProbeDataset 的简单 transform
    from dataset import LinearProbeDataset
    dataset = LinearProbeDataset(
        root=data_dir,
        train=False,            # 使用测试集图像
        download=False,
        num_samples=num_demo_images
    )

    print(f"  ✓ 加载了 {len(dataset)} 张无标签图像")
    print(f"  图像尺寸: 3 × 32 × 32 (CIFAR-10)")
    print()

    # ============================================================
    # Step 3: 输入图像 → 输出表征向量
    # ============================================================
    print("Step 3: 输入图像 → 输出表征向量")
    print("-" * 40)

    # 收集所有图像（忽略标签，因为这是无监督表征提取）
    all_images = []
    all_labels = []  # 仅用于后续分析，表征提取过程不需要标签
    for i in range(num_demo_images):
        img, label = dataset[i]
        all_images.append(img)
        all_labels.append(label)

    # 堆叠成 batch
    image_batch = torch.stack(all_images)  # [num_images, 3, 32, 32]

    print(f"  输入: {image_batch.shape}")
    print(f"    - 图像数量: {image_batch.shape[0]}")
    print(f"    - 通道数: {image_batch.shape[1]}")
    print(f"    - 高度: {image_batch.shape[2]}")
    print(f"    - 宽度: {image_batch.shape[3]}")

    # 核心调用：输入图像 → 输出表征向量
    features = extract_features(encoder, image_batch, device)

    print(f"  输出: {features.shape}")
    print(f"    - 图像数量: {features.shape[0]}")
    print(f"    - 表征维度: {features.shape[1]}")
    print(f"    - 每个图像被压缩为一个 {features.shape[1]} 维向量")
    print()

    # ============================================================
    # Step 4: 展示表征向量
    # ============================================================
    print("Step 4: 展示表征向量（前 5 维）")
    print("-" * 40)

    for i in range(min(5, num_demo_images)):
        vec = features[i].cpu().numpy()
        print(f"  图像 {i+1} 的表征向量前 5 维: "
              f"[{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}, {vec[3]:.4f}, {vec[4]:.4f}]")
    print(f"  ... (共 {feature_dim} 维)")
    print()

    # ============================================================
    # Step 5: 分析表征空间（相似图像应有相近的表征）
    # ============================================================
    print("Step 5: 表征空间分析")
    print("-" * 40)

    # 计算所有图像表征之间的余弦相似度矩阵
    features_norm = torch.nn.functional.normalize(features, dim=1)
    sim_matrix = torch.mm(features_norm, features_norm.t())  # [N, N]

    print(f"  余弦相似度矩阵形状: {sim_matrix.shape}")
    print(f"  相似度范围: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")

    # CIFAR-10 类别名称
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    print()
    print("  图像之间的余弦相似度（同类应高于异类）:")
    print(f"  {'图像对':<20} {'类别':<25} {'相似度':>10}")
    print(f"  {'-'*55}")

    for i in range(num_demo_images):
        for j in range(i + 1, num_demo_images):
            same_class = "✓ 同类" if all_labels[i] == all_labels[j] else "✗ 异类"
            pair_desc = f"图像{i+1} ↔ 图像{j+1}"
            class_desc = f"{class_names[all_labels[i]]} ↔ {class_names[all_labels[j]]} ({same_class})"
            sim = sim_matrix[i, j].item()
            print(f"  {pair_desc:<20} {class_desc:<25} {sim:>10.4f}")

    print()

    # ============================================================
    # Step 6: 总结
    # ============================================================
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print(f"""
    ┌─────────────────────────────────────────────┐
    │                                             │
    │   输入: {num_demo_images} 张无标签 CIFAR-10 图像          │
    │         (3 × 32 × 32 像素)                   │
    │                     ↓                       │
    │         SimCLR Encoder (预训练)              │
    │                     ↓                       │
    │   输出: {num_demo_images} 个 {feature_dim} 维图像表征向量      │
    │                                             │
    │   这就是 SimCLR 的核心能力:                   │
    │   无需标签，将图像映射为有语义意义的向量       │
    │   相似图像 → 相近向量 → 可用于分类/检索       │
    │                                             │
    └─────────────────────────────────────────────┘
    """)

    return features, all_labels, sim_matrix


def visualize_embedding_space(features, labels, save_path='../results/embedding_tsne.png'):
    """
    使用 t-SNE 可视化表征空间（加分项）。

    将高维表征向量降维到 2D，观察同类图像是否聚集在一起。

    参数：
        features (torch.Tensor): 表征向量，形状 [N, D]
        labels (list): 对应的类别标签
        save_path (str): 图片保存路径
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        # 降维到 2D
        features_np = features.cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(features_np) - 1))
        features_2d = tsne.fit_transform(features_np)

        # 绘图
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for class_idx in range(10):
            mask = np.array(labels) == class_idx
            if mask.sum() > 0:
                plt.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    c=[colors[class_idx]],
                    label=class_names[class_idx],
                    s=100,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )

        plt.title('t-SNE Visualization of SimCLR Image Representations', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nt-SNE 可视化已保存: {save_path}")

    except ImportError:
        print("\n⚠ 未安装 scikit-learn，跳过 t-SNE 可视化")
        print("  安装方法: pip install scikit-learn")


if __name__ == "__main__":
    # 运行演示
    features, labels, sim_matrix = demonstrate_pipeline()

    # 尝试 t-SNE 可视化（需要 scikit-learn）
    if len(features) >= 5:
        visualize_embedding_space(features, labels)