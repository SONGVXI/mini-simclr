"""
SimCLR 线性评估（Linear Probe）脚本。

在自监督预训练完成后，冻结 encoder，训练一个线性分类器。
评估流程：
1. 加载预训练的 encoder 权重
2. 冻结 encoder 所有参数
3. 在 encoder 后接一个线性分类器（512 -> 10）
4. 使用 CIFAR-10 标签训练线性分类器
5. 在测试集上评估分类准确率
6. 展示至少 5 张测试图像的预测结果
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from tqdm import tqdm

from dataset import LinearProbeDataset
from model import SmallCNNEncoder, LinearClassifier


def evaluate_encoder(model, classifier, test_loader, device):
    """
    在测试集上评估线性分类器准确率。

    参数：
        model (nn.Module): 冻结的 encoder
        classifier (nn.Module): 线性分类器
        test_loader (DataLoader): 测试数据加载器
        device (torch.device): 计算设备

    返回：
        tuple: (accuracy, predictions, ground_truth, images)
    """
    model.eval()
    classifier.eval()

    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # 冻结的 encoder 提取特征
            features = model(images)

            # 线性分类器预测
            logits = classifier(features)
            _, predicted = torch.max(logits, 1)

            # 统计正确数量
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集预测结果（用于可视化）
            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_images.append(images.cpu())

    accuracy = 100.0 * correct / total

    # 拼接所有图像
    all_images = torch.cat(all_images, dim=0)

    return accuracy, all_predictions, all_labels, all_images


def linear_probe(config):
    """
    执行线性评估（Linear Probe）。

    参数：
        config (dict): 评估配置字典，包含以下键：
            - encoder_path (str): 预训练 encoder 权重路径
            - feature_dim (int): encoder 输出维度
            - num_classes (int): 分类类别数
            - num_train_samples (int): 训练样本数
            - num_test_samples (int): 测试样本数
            - batch_size (int): 批大小
            - epochs (int): 线性分类器训练轮数
            - learning_rate (float): 学习率
            - device (str): 计算设备
            - data_dir (str): 数据集路径
            - save_dir (str): 结果保存路径

    返回：
        tuple: (accuracy, results_dict)
    """
    # 自动检测设备：优先使用配置的 device，CUDA 不可用时回退到 CPU
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("警告: 配置为 CUDA 但 CUDA 不可用，回退到 CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")

    # ============================================================
    # 1. 加载预训练的 encoder
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 1: 加载预训练 Encoder")
    print("=" * 60)

    encoder = SmallCNNEncoder(feature_dim=config['feature_dim']).to(device)

    if os.path.exists(config['encoder_path']):
        encoder.load_state_dict(torch.load(config['encoder_path'], map_location=device))
        print(f"成功加载预训练权重: {config['encoder_path']}")
    else:
        print(f"警告: 未找到预训练权重 {config['encoder_path']}，使用随机初始化权重")
        print("这将导致准确率接近随机猜测 (10%)")

    # 冻结 encoder：不计算梯度，不更新参数
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder 参数量: {encoder_params:,} (已冻结)")

    # ============================================================
    # 2. 初始化线性分类器
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: 初始化线性分类器")
    print("=" * 60)

    classifier = LinearClassifier(
        input_dim=config['feature_dim'],
        num_classes=config['num_classes']
    ).to(device)

    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"线性分类器参数量: {classifier_params:,}")

    # ============================================================
    # 3. 准备数据
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3: 准备线性评估数据集")
    print("=" * 60)

    # 训练集（用于训练线性分类器，使用标签）
    train_dataset = LinearProbeDataset(
        root='./data',
        train=True,
        download=True,
        num_samples=config['num_train_samples']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    # 测试集（用于评估准确率）
    test_dataset = LinearProbeDataset(
        root='./data',
        train=False,
        download=True,
        num_samples=config['num_test_samples']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    print(f"训练图像数量: {len(train_dataset)}")
    print(f"测试图像数量: {len(test_dataset)}")

    # ============================================================
    # 4. 训练线性分类器
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 4: 训练线性分类器")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config['learning_rate'])

    print(f"优化器: Adam (lr={config['learning_rate']})")
    print(f"损失函数: CrossEntropyLoss")

    train_loss_history = []
    train_acc_history = []

    for epoch in range(config['epochs']):
        encoder.eval()  # 确保 encoder 保持冻结
        classifier.train()

        epoch_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['epochs']}]")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # 冻结的 encoder 提取特征（不计算梯度）
            with torch.no_grad():
                features = encoder(images)

            # 线性分类器前向传播
            logits = classifier(features)
            loss = criterion(logits, labels)

            # 反向传播和优化（只更新分类器参数）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计准确率
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            epoch_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

        avg_loss = epoch_loss / num_batches
        train_accuracy = 100.0 * correct / total
        train_loss_history.append(avg_loss)
        train_acc_history.append(train_accuracy)

        print(f"Epoch [{epoch+1}/{config['epochs']}] - "
              f"Loss: {avg_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%")

    print("\n线性分类器训练完成！")

    # ============================================================
    # 5. 测试集评估
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 5: 测试集评估")
    print("=" * 60)

    test_accuracy, predictions, ground_truth, test_images = evaluate_encoder(
        encoder, classifier, test_loader, device
    )

    print(f"\n{'='*60}")
    print(f"测试集准确率: {test_accuracy:.2f}%")
    print(f"{'='*60}")

    # ============================================================
    # 6. 保存结果
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 6: 保存评估结果")
    print("=" * 60)

    os.makedirs(config['save_dir'], exist_ok=True)

    # 保存分类器权重
    classifier_path = os.path.join(config['save_dir'], 'linear_classifier.pth')
    torch.save(classifier.state_dict(), classifier_path)
    print(f"线性分类器权重已保存: {classifier_path}")

    # 组装结果
    results = {
        'test_accuracy': test_accuracy,
        'num_train_samples': config['num_train_samples'],
        'num_test_samples': config['num_test_samples'],
        'pretrain_epochs': config.get('pretrain_epochs', 'N/A'),
        'linear_probe_epochs': config['epochs'],
        'train_loss_history': train_loss_history,
        'train_acc_history': train_acc_history,
        'predictions': predictions[:20],     # 保存前 20 个预测结果
        'ground_truth': ground_truth[:20],   # 保存前 20 个真实标签
    }

    # 保存结果 JSON
    results_path = os.path.join(config['save_dir'], 'linear_probe_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"评估结果已保存: {results_path}")

    return test_accuracy, results, test_images, predictions, ground_truth


def visualize_predictions(test_images, predictions, ground_truth, num_samples=5, save_path=None):
    """
    可视化测试图像的预测结果，展示真实类别和预测类别。

    参数：
        test_images (torch.Tensor): 测试图像张量
        predictions (list): 模型预测类别列表
        ground_truth (list): 真实类别列表
        num_samples (int): 展示的图像数量
        save_path (str): 图片保存路径
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # CIFAR-10 类别名称
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # CIFAR-10 标准化参数（用于反标准化）
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 4))

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        idx = np.random.randint(0, len(test_images))
        img = test_images[idx].numpy()

        # 反标准化：img = img * std + mean
        img = img.transpose(1, 2, 0)  # CHW -> HWC
        img = img * std + mean
        img = np.clip(img, 0, 1)

        true_label = ground_truth[idx]
        pred_label = predictions[idx]

        axes[i].imshow(img)
        axes[i].set_title(
            f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}',
            fontsize=10,
            color='green' if true_label == pred_label else 'red'
        )
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"预测结果可视化已保存: {save_path}")

    plt.close()


def plot_linear_probe_curves(train_loss, train_acc, save_path):
    """
    绘制线性评估阶段的训练损失和准确率曲线。

    参数：
        train_loss (list): 训练损失历史
        train_acc (list): 训练准确率历史
        save_path (str): 图片保存路径
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_loss) + 1)

    # 损失曲线
    ax1.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Linear Probe Training Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, train_acc, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Linear Probe Training Accuracy', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"线性评估曲线已保存: {save_path}")


if __name__ == "__main__":
    # ============================================================
    # 线性评估配置
    # ============================================================
    config = {
        # 模型配置
        'encoder_path': './results/simclr_encoder.pth',  # 预训练 encoder 权重路径
        'feature_dim': 512,                                 # encoder 输出维度
        'num_classes': 10,                                  # CIFAR-10 类别数

        # 数据配置
        'data_dir': './data',                              # 数据集路径
        'num_train_samples': 5000,                          # 线性评估训练样本数（带标签）
        'num_test_samples': 2000,                           # 测试样本数
        'batch_size': 128,                                  # 批大小（GPU 可用更大 batch）

        # 训练配置
        'epochs': 10,                                       # 线性分类器训练轮数
        'learning_rate': 1e-3,                              # Adam 学习率
        'device': 'cuda',                                   # 使用 GPU 训练（RTX 4060）

        # 保存配置
        'save_dir': './results',                           # 结果保存路径
        'pretrain_epochs': 10,                              # 预训练 epoch 数（用于记录）
    }

    print("=" * 60)
    print("Mini-SimCLR 线性评估 (Linear Probe)")
    print("=" * 60)
    print(f"配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # 执行线性评估
    test_accuracy, results, test_images, predictions, ground_truth = linear_probe(config)

    # 绘制训练曲线
    curves_path = os.path.join(config['save_dir'], 'linear_probe_curves.png')
    plot_linear_probe_curves(
        results['train_loss_history'],
        results['train_acc_history'],
        curves_path
    )

    # 可视化预测结果
    vis_path = os.path.join(config['save_dir'], 'prediction_samples.png')
    visualize_predictions(test_images, predictions, ground_truth, num_samples=5, save_path=vis_path)

    # 打印最终结果总结
    print("\n" + "=" * 60)
    print("线性评估结果总结")
    print("=" * 60)
    print(f"使用训练图像数: {config['num_train_samples']}")
    print(f"预训练 Epoch: {config['pretrain_epochs']}")
    print(f"线性评估 Epoch: {config['epochs']}")
    print(f"测试准确率: {test_accuracy:.2f}%")
    print(f"预测结果可视化: {vis_path}")
    print(f"训练曲线: {curves_path}")
    print("=" * 60)

    # 分析准确率
    if test_accuracy < 15.0:
        print("\n⚠️ 警告: 准确率接近随机猜测 (10%)")
        print("可能原因:")
        print("  1. 预训练 epoch 太少，encoder 未学到有效特征")
        print("  2. 训练样本数太少，对比学习不够充分")
        print("  3. batch size 太小，负样本不够丰富")
        print("  4. 数据增强过强，破坏了语义信息")
        print("  5. encoder 太浅，表征能力不足")
        print("建议: 增加预训练 epoch 或训练样本数")
    elif test_accuracy < 30.0:
        print("\n📊 准确率有所提升，但仍较低")
        print("建议: 增加预训练 epoch 或使用更多训练样本")
    else:
        print(f"\n✅ 线性评估准确率达到 {test_accuracy:.2f}%，模型学到了有效的图像表征")