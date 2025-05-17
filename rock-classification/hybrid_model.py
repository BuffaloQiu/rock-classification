import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from PIL import Image
import pickle
import logging
from tqdm import tqdm


# ---------------------- 特征计算模块 ----------------------
def compute_traditional_features(image_path, cache_dir="feature_cache"):
    """计算颜色直方图+纹理特征（52维），并支持缓存机制"""
    # 创建缓存目录结构
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # 基于图像路径生成缓存文件名
        rel_path = os.path.relpath(image_path, start=os.path.commonpath(config["data_dirs"]))
        cache_path = os.path.join(cache_dir, os.path.splitext(rel_path)[0] + ".pkl")

        # 检查缓存是否存在
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"加载缓存失败: {cache_path}, 重新计算特征. 错误: {e}")

    # 特征计算逻辑
    try:
        image = Image.open(image_path).convert('RGB')
        img_np = np.array(image)

        # 颜色直方图（16 bins × 3通道）
        color_features = []
        for channel in range(3):
            hist = np.histogram(img_np[:, :, channel], bins=16, range=(0, 255))[0]
            hist = hist / (hist.sum() + 1e-6)  # 防止除零
            color_features.extend(hist)

        # 灰度共生矩阵纹理特征
        gray_img = rgb2gray(img_np)
        gray_img = (gray_img * 255).astype(np.uint8)
        glcm = graycomatrix(gray_img, distances=[1],
                            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=256, symmetric=True, normed=True)
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))

        features = np.concatenate([color_features, [contrast, correlation, energy, homogeneity]])

        # 保存缓存
        if cache_dir:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(features, f)

        return features

    except Exception as e:
        logging.error(f"特征计算失败: {image_path}, 错误: {str(e)}")
        return None


# ---------------------- 数据集定义 ----------------------
class HybridDataset(Dataset):
    def __init__(self, data_dirs, class_names, transform=None, cache_dir="feature_cache"):
        self.data = []
        self.transform = transform
        self.cache_dir = cache_dir

        # 遍历所有类别
        for class_idx, (data_dir, class_name) in enumerate(zip(data_dirs, class_names)):
            if not os.path.exists(data_dir):
                logging.warning(f"数据目录不存在: {data_dir}")
                continue

            logging.info(f"正在加载类别: {class_name}")
            for root, _, files in os.walk(data_dir):
                for file in tqdm(files, desc=f"处理 {class_name} 文件"):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)
                        features = compute_traditional_features(image_path, self.cache_dir)

                        if features is not None:
                            self.data.append({
                                'image_path': image_path,
                                'features': features,
                                'label': class_idx,
                                'class_name': class_name
                            })

        # 打乱数据集顺序
        np.random.seed(42)
        np.random.shuffle(self.data)

        logging.info(f"数据集加载完成，总样本数: {len(self)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(item['features'], dtype=torch.float32), item['label']


# ---------------------- 混合模型定义 ----------------------
class HybridModel(nn.Module):
    def __init__(self, cnn_model, feature_size=52, num_classes=3):
        super().__init__()
        # 使用预训练的ResNet作为CNN特征提取器
        self.cnn = cnn_model
        self.cnn.fc = nn.Identity()  # 移除原全连接层

        # 传统特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 混合特征分类器
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 256),  # ResNet输出512维 + 传统特征128维
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, features):
        cnn_features = self.cnn(image)
        encoded_features = self.feature_encoder(features)
        combined = torch.cat([cnn_features, encoded_features], dim=1)
        return self.fusion(combined)


# ---------------------- 早停机制 ----------------------
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        if self.verbose:
            logging.info(f'验证准确率提高 ({self.val_loss_min:.6f} --> {val_acc:.6f}). 保存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_acc


# ---------------------- 训练流程 ----------------------
def train_hybrid_model():
    # 配置参数
    global config  # 使compute_traditional_features能访问配置
    config = {
        "data_dirs": [
            r"D:\BaiduNetdiskDownload\变质岩\南京大学变质岩教学薄片照片数据集",
            r"D:\BaiduNetdiskDownload\沉积岩",
            r"D:\BaiduNetdiskDownload\火成岩\南京大学火成岩教学薄片照片数据集"
        ],
        "class_names": ["变质岩", "沉积岩", "火成岩"],
        "batch_size": 32,
        "num_epochs": 100,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "cache_dir": "feature_cache",
        "model_save_path": "hybrid_model.pth"
    }

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )

    # 数据预处理
    transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val_test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 创建数据集
    train_dataset = HybridDataset(config["data_dirs"], config["class_names"],
                                  transform=transform["train"], cache_dir=config["cache_dir"])

    # 划分数据集
    train_size = int(0.8 * len(train_dataset))
    val_size = int(0.1 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 为验证集和测试集设置正确的transform
    val_dataset.dataset.transform = transform["val_test"]
    test_dataset.dataset.transform = transform["val_test"]

    logging.info(f"训练集样本数: {len(train_dataset)}")
    logging.info(f"验证集样本数: {len(val_dataset)}")
    logging.info(f"测试集样本数: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4,
                             pin_memory=True)

    # 初始化模型
    cnn_backbone = models.resnet18(pretrained=True)
    model = HybridModel(cnn_backbone, feature_size=52, num_classes=len(config["class_names"]))

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    if torch.cuda.device_count() > 1:
        logging.info(f"使用 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)

    model = model.to(device)

    # 定义优化器与损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=10, verbose=True, path=config["model_save_path"])

    # 训练循环
    best_acc = 0.0
    for epoch in range(config["num_epochs"]):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f'Epoch {epoch + 1}/{config["num_epochs"]}')
        for i, (images, features, labels) in progress_bar:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item(), 'acc': train_correct / train_total})

        train_loss = running_loss / len(train_dataset)
        train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, features, labels in val_loader:
                images = images.to(device)
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(images, features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / len(val_dataset)

        # 学习率调整
        scheduler.step(val_acc)

        # 早停检查
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            logging.info("早停触发!")
            break

        # 记录日志
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']} | "
                     f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                     f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 测试集评估
    model.load_state_dict(torch.load(config["model_save_path"]))
    model.eval()
    test_correct = 0
    class_correct = list(0. for i in range(len(config["class_names"])))
    class_total = list(0. for i in range(len(config["class_names"])))

    with torch.no_grad():
        for images, features, labels in test_loader:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(images, features)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()

            # 按类别统计准确率
            c = (preds == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    test_acc = test_correct / len(test_dataset)
    logging.info(f"测试集准确率: {test_acc:.4f}")

    # 打印各类别准确率
    for i, class_name in enumerate(config["class_names"]):
        if class_total[i] > 0:
            logging.info(f'{class_name} 准确率: {100 * class_correct[i] / class_total[i]:.2f}%')
        else:
            logging.info(f'{class_name} 无样本')

    # 保存完整模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': config["class_names"]
    }, config["model_save_path"].replace('.pth', '_full.pth'))


if __name__ == "__main__":
    train_hybrid_model()    