# -*- coding: utf-8 -*-
"""
U-Net 批量预测脚本（Oxford Pets 多类分割）
输入：test_imgs 文件夹中的图片
输出：
  - test_results_paired/          （可视化结果：拼合图 + 索引图）
  - test_results_deliverables/     （交付数据：每张图的 JSON 统计 + 汇总 CSV）
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import json
import csv
from collections import Counter

# -------------------- 1. 定义模型结构（与训练时一致）--------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# -------------------- 2. 加载模型和配置 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 37  # Oxford Pets 数据集：背景 + 36种宠物
model = UNet(n_channels=3, n_classes=num_classes).to(device)

model_path = 'best_unet_oxford_multiclass.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练或修改路径。")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"模型加载成功，使用设备：{device}")

image_size = (256, 256)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])


# -------------------- 3. 颜色映射（tab20+tab20b 组合）--------------------
tab20_colors = cm.tab20(range(20))
tab20b_colors = cm.tab20b(range(20))
all_colors = list(tab20_colors) + list(tab20b_colors)
colors = all_colors[:num_classes]
cmap = mcolors.ListedColormap(colors)

def mask_to_colored(mask_np):
    colored = cmap(mask_np)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return colored


# -------------------- 4. 准备输出文件夹 --------------------
input_folder = 'test_imgs'

# 可视化结果文件夹（原拼合图和索引图）
vis_folder = 'test_results_paired'
os.makedirs(vis_folder, exist_ok=True)

# 交付数据文件夹（统计 JSON 和 CSV）
deliver_folder = 'test_results_deliverables'
os.makedirs(deliver_folder, exist_ok=True)

valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
image_files.sort()
print(f"找到 {len(image_files)} 张图片，开始处理...\n")

# 准备汇总数据列表（用于 CSV）
summary_data = []

# -------------------- 5. 批量处理 --------------------
for img_file in image_files:
    img_path = os.path.join(input_folder, img_file)
    try:
        # 读取原图
        img_pil = Image.open(img_path).convert('RGB')
        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (H, W)

        # 生成彩色掩码和拼合图（可视化用）
        colored_mask = mask_to_colored(pred_mask)
        img_resized = img_pil.resize(image_size, Image.Resampling.LANCZOS)
        img_resized_np = np.array(img_resized)
        paired = np.hstack((img_resized_np, colored_mask))
        paired_img = Image.fromarray(paired)

        base_name = os.path.splitext(img_file)[0]

        # ---------- 保存可视化结果到 vis_folder ----------
        paired_path = os.path.join(vis_folder, f"{base_name}_paired.png")
        paired_img.save(paired_path)

        index_img = Image.fromarray(pred_mask.astype(np.uint8), mode='L')
        index_path = os.path.join(vis_folder, f"{base_name}_index.png")
        index_img.save(index_path)

        # ---------- 生成统计信息（交付数据） ----------
        class_counts = Counter(pred_mask.flatten())
        total_pixels = pred_mask.size

        # 构建统计字典
        stats = {
            'image_file': img_file,
            'total_pixels': total_pixels,
            'class_pixel_counts': {int(k): int(v) for k, v in class_counts.items()},
            'class_ratios': {int(k): float(v / total_pixels) for k, v in class_counts.items()}
        }

        # 保存 JSON 到 deliver_folder
        json_path = os.path.join(deliver_folder, f"{base_name}_stats.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # 添加到汇总列表（用于 CSV）
        row = {'image_file': img_file, 'total_pixels': total_pixels}
        for cls in range(num_classes):
            count = class_counts.get(cls, 0)
            row[f'class_{cls}_pixels'] = count
            row[f'class_{cls}_ratio'] = count / total_pixels
        summary_data.append(row)

        print(f"✓ {img_file} -> 可视化已保存到 {vis_folder}，统计数据已保存到 {deliver_folder}")

    except Exception as e:
        print(f"❌ {img_file} 处理失败: {e}\n")

# -------------------- 6. 写入汇总 CSV 到 deliver_folder --------------------
if summary_data:
    csv_path = os.path.join(deliver_folder, 'summary.csv')
    fieldnames = list(summary_data[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"\n汇总 CSV 已保存至 {csv_path}")
else:
    print("\n没有成功处理的图片，未生成 CSV。")

print(f"\n所有处理完成！")
print(f"  - 可视化结果：{vis_folder}/")
print(f"  - 交付数据：{deliver_folder}/")