import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 固定随机种子保证可复现性
torch.manual_seed(42)
np.random.seed(42)


# ---------------------- 数据预处理 ----------------------
def get_data_loaders(data_root, batch_size=32):
    """
    创建数据加载器，自动划分训练集、验证集和测试集
    """
    # 定义训练集和测试集的不同预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 加载数据集
    full_dataset = datasets.ImageFolder(root=data_root)

    # 自动统计类别信息
    class_names = full_dataset.classes
    print(f"发现 {len(class_names)} 个类别: {class_names}")

    # 创建类别索引映射
    class_to_idx = full_dataset.class_to_idx

    # 划分数据集（8:1:1）
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 为不同数据集分配不同的预处理
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_names, class_to_idx


# ---------------------- 模型定义 ----------------------
class SedimentaryClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # 使用预训练的ResNet50模型
        self.base_model = models.resnet50(pretrained=True)

        # 冻结大部分预训练层，只训练最后几层
        for param in list(self.base_model.parameters())[:-10]:
            param.requires_grad = False

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


# ---------------------- 训练函数 ----------------------
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=15, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    best_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        # 每个epoch的训练和验证阶段
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化仅在训练阶段进行
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # 统计指标
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 学习率调度
            if phase == "train" and scheduler:
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # 记录历史数据
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 保存最佳模型
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch + 1
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": epoch_acc,
                }, "sedimentary_model_best.pth")
                print(f"保存新的最佳模型，准确率: {best_acc:.4f}")

    # 加载最佳模型
    checkpoint = torch.load("sedimentary_model_best.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"训练完成，最佳模型在第 {best_epoch} 个epoch，准确率: {best_acc:.4f}")

    return model, history


# ---------------------- 评估模型 ----------------------
def evaluate_model(model, test_loader, class_names, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"\n测试集准确率: {accuracy:.2f}%")

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("混淆矩阵")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    return accuracy, cm


# ---------------------- 可视化训练历史 ----------------------
def visualize_training(history):
    plt.figure(figsize=(14, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="训练损失")
    plt.plot(history["val_loss"], label="验证损失")
    plt.title("训练和验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="训练准确率")
    plt.plot(history["val_acc"], label="验证准确率")
    plt.title("训练和验证准确率")
    plt.xlabel("Epoch")
    plt.ylabel("准确率")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()


# ---------------------- 预测单张图像 ----------------------
def predict_single_image(model, image_path, class_names, class_to_idx, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        # 预测
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = class_names[predicted_idx.item()]

        # 获取索引到类别的映射
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        print(f"\n预测结果: {predicted_class} (类别索引: {predicted_idx.item()})")
        print(f"置信度: {confidence.item() * 100:.2f}%")

        # 显示前5个预测结果
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        print("\n前5个可能的类别:")
        for i in range(top5_probs.size(1)):
            class_name = class_names[top5_indices[0, i].item()]
            prob = top5_probs[0, i].item() * 100
            print(f"{i + 1}. {class_name}: {prob:.2f}%")

        # 可视化预测结果
        plt.figure(figsize=(12, 5))

        # 显示图像
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image))
        plt.title(f"预测结果: {predicted_class}\n置信度: {confidence.item() * 100:.2f}%")
        plt.axis("off")

        # 显示概率分布
        plt.subplot(1, 2, 2)
        plt.barh([class_names[i] for i in top5_indices[0].cpu().numpy()],
                 top5_probs[0].cpu().numpy())
        plt.xlabel("概率")
        plt.title("前5个预测类别")
        plt.tight_layout()
        plt.savefig("prediction_result.png")
        plt.close()

        return predicted_class, confidence.item(), idx_to_class

    except Exception as e:
        print(f"预测时出错: {e}")
        return None, None, None


# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # 配置参数
    config = {
        "data_root": r"D:\BaiduNetdiskDownload\沉积岩",  # 数据集路径
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "lr_step_size": 20,
        "lr_gamma": 0.5
    }

    # 获取数据加载器
    train_loader, val_loader, test_loader, class_names, class_to_idx = get_data_loaders(
        config["data_root"],
        batch_size=config["batch_size"]
    )

    # 初始化模型
    model = SedimentaryClassifier(num_classes=len(class_names))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr_step_size"],
        gamma=config["lr_gamma"]
    )

    # 组合数据加载器字典
    dataloaders_dict = {
        "train": train_loader,
        "val": val_loader
    }

    # 开始训练
    print("开始训练模型...")
    trained_model, history = train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        scheduler,
        num_epochs=config["num_epochs"]
    )

    # 可视化训练历史
    visualize_training(history)
    print("训练历史已保存为 training_history.png")

    # 在测试集上评估
    print("\n在测试集上评估模型...")
    accuracy, cm = evaluate_model(trained_model, test_loader, class_names)
    print("混淆矩阵已保存为 confusion_matrix.png")

