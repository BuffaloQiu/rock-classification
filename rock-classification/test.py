import os
import pandas as pd
from torchvision import models
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray


class HybridModel(nn.Module):
    def __init__(self, cnn_model, feature_size=52, num_classes=3):
        super().__init__()
        self.cnn = cnn_model
        self.cnn.fc = nn.Identity()  # 移除原全连接层

        # 修正后的特征编码器（Linear -> BatchNorm -> ReLU，匹配预训练模型结构）
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_size, 128),          # 0: Linear
            nn.BatchNorm1d(128),                    # 1: BatchNorm（对应预训练模型的feature_encoder.1）
            nn.ReLU()                               # 2: ReLU（对应预训练模型的feature_encoder.2）
        )

        # 修正后的融合层（Linear -> BatchNorm -> ReLU -> Dropout -> Linear）
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 256),             # 0: Linear
            nn.BatchNorm1d(256),                    # 1: BatchNorm（对应预训练模型的fusion.1）
            nn.ReLU(),                               # 2: ReLU（对应预训练模型的fusion.2）
            nn.Dropout(0.3),                         # 3: Dropout
            nn.Linear(256, num_classes)              # 4: Linear
        )

    def forward(self, image, features):
        cnn_features = self.cnn(image)
        traditional_features = self.feature_encoder(features)
        combined = torch.cat([cnn_features, traditional_features], dim=1)
        return self.fusion(combined)


def compute_traditional_features(image_path):
    """计算颜色直方图+纹理特征（52维）"""
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
    glcm = graycomatrix(gray_img,
                        distances=[1],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256,
                        symmetric=True,
                        normed=True)
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))

    return np.concatenate([color_features, [contrast, correlation, energy, homogeneity]])


# ---------------------- 加载模型 ----------------------
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 获取设备
    # 加载三岩类分类模型
    cnn_backbone = models.resnet18(weights=None)  # 移除pretrained警告
    hybrid_model = HybridModel(cnn_backbone, feature_size=52, num_classes=3)
    hybrid_model.load_state_dict(
        torch.load("hybrid_model.pth", map_location=device)  # 统一设备
    )

    # 加载沉积岩子类模型（假设其结构正确，若有类似问题需同步修正）
    class SedimentaryClassifier(nn.Module):
        def __init__(self, num_classes=9):
            super().__init__()
            self.base_model = models.resnet18(weights=None)
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            return self.base_model(x)

    sedimentary_model = SedimentaryClassifier(num_classes=9)
    sedimentary_model.load_state_dict(
        torch.load("sedimentary_model.pth", map_location=device)
    )

    return hybrid_model.eval(), sedimentary_model.eval()


# ---------------------- 分类流程 ----------------------
def classify_attachment1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid_model, sedimentary_model = load_models()
    hybrid_model = hybrid_model.to(device)
    sedimentary_model = sedimentary_model.to(device)

    # 定义预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 读取模板文件（路径修正为原始字符串）
    excel_path = r'D:\BaiduNetdiskDownload\赛题提交结果.xlsx'
    df = pd.read_excel(excel_path)
    results = []

    # 类别标签映射
    main_classes = ["变质岩", "沉积岩", "火成岩"]
    sedimentary_subclasses = [
        '火山碎屑岩', '砂岩', '泥页岩', '粉砂岩',
        '灰岩', '白云岩', '硅质岩', '蒸发岩', '其他'
    ]

    for idx, row in df.iterrows():
        image_name = row["图片名称"]
        image_path = os.path.join(r"D:\BaiduNetdiskDownload\附件1", image_name)  # 路径修正为原始字符串

        if not os.path.exists(image_path):
            results.append({"图片名称": image_name, "岩石类别": "未找到文件"})
            continue

        try:
            # Step 1: 计算传统特征
            features = compute_traditional_features(image_path)
            features_tensor = torch.tensor(features).unsqueeze(0).float().to(device)

            # Step 2: 三岩类分类
            image = Image.open(image_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                main_output = hybrid_model(img_tensor, features_tensor)
                main_class_idx = main_output.argmax().item()
                main_class = main_classes[main_class_idx]

                # Step 3: 沉积岩细分类
                if main_class == "沉积岩":
                    sub_output = sedimentary_model(img_tensor)
                    sub_class_idx = sub_output.argmax().item()
                    sub_class = sedimentary_subclasses[sub_class_idx]

                    # 过滤非目标子类
                    if sub_class in ['砂岩', '泥页岩', '粉砂岩', '灰岩', '白云岩']:
                        final_class = sub_class
                    else:
                        final_class = "沉积岩（其他）"
                else:
                    final_class = main_class

                results.append({"图片名称": image_name, "岩石类别": final_class})

        except Exception as e:
            print(f"处理失败: {image_name}, {str(e)}")
            results.append({"图片名称": image_name, "岩石类别": "分类错误"})

    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_excel("bdc241158.xlsx", index=False)
    print("分类完成，结果已保存至 bdc241158.xlsx")

    # 统计类别分布
    print("\n分类统计结果:")
    print(result_df["岩石类别"].value_counts())


if __name__ == "__main__":
    classify_attachment1()