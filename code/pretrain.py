"""
SimCLR 自监督预训练脚本。

使用无标签的 CIFAR-10 图像进行对比学习预训练。
训练过程：
1. 加载 SimCLR 数据集（双视图增强）
2. 初始化 SimCLR 模型（encoder + projection head）
3. 使用 NT-Xent 损失进行对比学习
4. 每个 epoch 记录 contrastive loss
5. 保存预训练后的 encoder 权重
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm

from dataset import SimCLRDataset
from model import SimCLRModel
from loss import NTXentLoss


def pretrain_simclr(config):
    """
    执行 SimCLR 自监督预训练。

    参数：
        config (dict): 训练配置字典，包含以下键：
            - num_samples (int): 使用的训练图像数量
            - batch_size (int): 批大小
            - epochs (int): 训练轮数
            - learning_rate (float): 学习率
            - temperature (float): NT-Xent 损失的温度参数
            - feature_dim (int): encoder 输出维度
            - proj_output_dim (int): projection head 输出维度
            - device (str): 训练设备 ('cpu' 或 'cuda')
            - save_dir (str): 模型保存路径
            - data_dir (str): 数据集路径

    返回：
        tuple: (model, loss_history) 训练好的模型和损失历史记录
    """
    # ============================================================
    # 1. 准备数据
    # ============================================================
    print("=" * 60)
    print("Step 1: 加载 SimCLR 数据集（双视图增强）")
    print("=" * 60)

    train_dataset = SimCLRDataset(
        root = './data',
        train=True,
        download=True,
        num_samples=config['num_samples'],
        image_size=32
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,      # CPU 训练使用 0，避免多进程问题
        drop_last=True      # 丢弃最后不完整的 batch，避免 batch size 不一致
    )

    print(f"训练图像数量: {len(train_dataset)}")
    print(f"Batch 数量: {len(train_loader)}")
    print(f"Batch Size: {config['batch_size']}")

    # ============================================================
    # 2. 初始化模型
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: 初始化 SimCLR 模型")
    print("=" * 60)

    # 自动检测设备：优先使用配置的 device，CUDA 不可用时回退到 CPU
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("警告: 配置为 CUDA 但 CUDA 不可用，回退到 CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model = SimCLRModel(
        feature_dim=config['feature_dim'],
        proj_hidden_dim=config['proj_hidden_dim'],
        proj_output_dim=config['proj_output_dim']
    ).to(device)

    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # ============================================================
    # 3. 初始化损失函数和优化器
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3: 初始化损失函数和优化器")
    print("=" * 60)

    criterion = NTXentLoss(temperature=config['temperature'])

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-6
    )

    # 学习率调度器：余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )

    print(f"优化器: Adam (lr={config['learning_rate']}, weight_decay=1e-6)")
    print(f"损失函数: NT-Xent (temperature={config['temperature']})")
    print(f"学习率调度: CosineAnnealingLR (T_max={config['epochs']})")

    # ============================================================
    # 4. 训练循环
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 4: 开始自监督预训练")
    print("=" * 60)

    loss_history = []  # 记录每个 epoch 的平均损失

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # 使用 tqdm 显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['epochs']}]")

        for view1, view2 in progress_bar:
            # 将数据移动到设备
            view1 = view1.to(device)
            view2 = view2.to(device)

            # 前向传播：获取两个视图的投影向量
            # view1 和 view2 是同一张图像的两个不同增强
            z1 = model(view1)  # 投影向量，形状 [batch_size, proj_output_dim]
            z2 = model(view2)  # 投影向量，形状 [batch_size, proj_output_dim]

            # 计算 NT-Xent 对比损失
            loss = criterion(z1, z2)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            epoch_loss += loss.item()
            num_batches += 1

            # 更新进度条显示
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        # 更新学习率
        scheduler.step()

        # 计算并记录 epoch 平均损失
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        print(f"Epoch [{epoch+1}/{config['epochs']}] - "
              f"Average Loss: {avg_loss:.4f}, "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    print("\n自监督预训练完成！")

    # ============================================================
    # 5. 保存模型和训练记录
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 5: 保存模型和训练记录")
    print("=" * 60)

    os.makedirs(config['save_dir'], exist_ok=True)

    # 保存 encoder 权重（用于后续 linear probe）
    encoder_path = os.path.join(config['save_dir'], 'simclr_encoder.pth')
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"Encoder 权重已保存: {encoder_path}")

    # 保存完整模型（含 projection head）
    model_path = os.path.join(config['save_dir'], 'simclr_full_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"完整模型已保存: {model_path}")

    # 保存损失记录
    loss_path = os.path.join(config['save_dir'], 'loss_history.json')
    with open(loss_path, 'w', encoding='utf-8') as f:
        json.dump({
            'loss_history': loss_history,
            'config': {k: str(v) if not isinstance(v, (int, float, bool, list, type(None))) else v
                       for k, v in config.items()}
        }, f, indent=2, ensure_ascii=False)
    print(f"损失记录已保存: {loss_path}")

    # 保存训练配置
    config_path = os.path.join(config['save_dir'], 'pretrain_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, bool, list, type(None))) else v
                   for k, v in config.items()}, f, indent=2, ensure_ascii=False)
    print(f"训练配置已保存: {config_path}")

    return model, loss_history


def plot_loss_curve(loss_history, save_path):
    """
    绘制并保存损失曲线。

    参数：
        loss_history (list): 每个 epoch 的损失值列表
        save_path (str): 图片保存路径
    """
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(loss_history) + 1)

    plt.plot(epochs, loss_history, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('NT-Xent Loss', fontsize=14)
    plt.title('SimCLR Pretraining Loss Curve', fontsize=16)
    plt.grid(True, alpha=0.3)

    # 标注每个点的损失值
    for i, loss in enumerate(loss_history):
        plt.annotate(f'{loss:.4f}', (i + 1, loss),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"损失曲线已保存: {save_path}")


if __name__ == "__main__":
    # ============================================================
    # 训练配置
    # ============================================================
    # GPU 配置：使用 5000 张图像，小型 CNN，RTX 4060 可轻松运行
    # 如显存不足可降低 batch_size 或 num_samples
    config = {
        # 数据配置
        'data_dir': './data',          # CIFAR-10 数据集路径
        'num_samples': 5000,            # 使用 5000 张训练图像
        'batch_size': 128,              # 批大小（GPU 可用更大 batch）

        # 模型配置
        'feature_dim': 512,             # encoder 输出维度
        'proj_hidden_dim': 512,         # projection head 隐藏层维度
        'proj_output_dim': 128,         # projection head 输出维度

        # 训练配置
        'epochs': 10,                   # 训练轮数（GPU 可跑更多 epoch）
        'learning_rate': 1e-3,          # Adam 学习率
        'temperature': 0.5,             # NT-Xent 温度参数
        'device': 'cuda',               # 使用 GPU 训练（RTX 4060）

        # 保存配置
        'save_dir': './results',       # 模型和结果保存路径
    }

    print("=" * 60)
    print("Mini-SimCLR 自监督预训练")
    print("=" * 60)
    print(f"配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # 执行预训练
    model, loss_history = pretrain_simclr(config)

    # 绘制损失曲线
    loss_curve_path = os.path.join(config['save_dir'], 'loss_curve.png')
    plot_loss_curve(loss_history, loss_curve_path)

    # 打印最终结果
    print("\n" + "=" * 60)
    print("预训练结果总结")
    print("=" * 60)
    print(f"训练图像数: {config['num_samples']}")
    print(f"预训练 Epoch: {config['epochs']}")
    print(f"初始 Loss: {loss_history[0]:.4f}")
    print(f"最终 Loss: {loss_history[-1]:.4f}")
    print(f"Loss 下降: {loss_history[0] - loss_history[-1]:.4f}")
    print(f"损失曲线: {loss_curve_path}")
    print(f"模型保存: {config['save_dir']}/simclr_encoder.pth")
    print("=" * 60)