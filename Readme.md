# U-Net 语义分割实践记录 —— Oxford Pets 多类分割

> 本项目是我学习 U-Net 语义分割的完整练习记录，使用 Oxford-IIIT Pet 数据集（37 类）进行训练，并编写了批量预测脚本，最终生成可视化和标准化的交付数据。目的是为后续的无人机藻类识别项目打下坚实基础。

---

## 📌 项目目标

- 掌握 U‑Net 的完整使用流程：数据加载、模型构建、训练、评估、预测。
- 理解语义分割的核心概念（像素级分类）及其与分类 CNN 的区别。
- 学会从预测结果中提取量化信息（各类别像素数、占比），为后续模块（数模组）提供标准输入。
- 培养解决实际问题的能力：环境配置、维度错误、颜色映射、数据交付格式等。

---

## 🛠️ 环境配置

| 组件          | 版本 / 详情                          |
| ------------- | ------------------------------------ |
| 操作系统      | Windows 10                           |
| Python        | 3.12                                  |
| 包管理器      | Conda（环境路径 `E:\conda_env\unet_practice`） |
| PyTorch       | 2.2.2 + CUDA 11.8（实际也可用 2.10.0+cu128） |
| torchvision   | 0.17.2                                |
| 其他库        | matplotlib, numpy, tqdm, pillow, opencv-python |
| 显卡          | RTX 5060（支持 CUDA 12.8+）           |

**创建环境命令**：
```bash
conda create --prefix E:\conda_env\unet_practice python=3.12 -y
conda activate E:\conda_env\unet_practice
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy tqdm pillow opencv-python
```

---

## 📂 数据集

使用 `torchvision.datasets.OxfordIIITPet`，该数据集包含 37 类（背景 + 36 种宠物），图像为 RGB，掩码为像素级标签（0~36）。数据集自动下载到 `./data` 目录。

**预处理**：
- 图像缩放至 `256×256`（保留长宽比，用 `Resize`）。
- 掩码缩放使用最近邻插值，保持标签整数。
- 训练/验证划分：80% 训练（2944 张），20% 验证（736 张）。

---

## 🧠 模型

实现标准 U‑Net 架构：
- 编码器：4 次下采样（64→128→256→512→1024）
- 解码器：4 次上采样 + 跳跃连接
- 输出通道：37（对应 37 个类别）
- 损失函数：`nn.CrossEntropyLoss()`
- 优化器：Adam（lr=1e-3）
- 评估指标：mIoU（平均交并比）

训练 20 个 epoch，最终验证集 mIoU ≈ **0.72**。

**训练日志片段**：
```
Epoch 1: Train Loss=0.8419, Val Loss=0.5381, Val mIoU=0.5420
Epoch 5: Train Loss=0.4095, Val Loss=0.3661, Val mIoU=0.6303
Epoch 10: Train Loss=0.3119, Val Loss=0.3283, Val mIoU=0.6709
Epoch 15: Train Loss=0.2429, Val Loss=0.3059, Val mIoU=0.7025
Epoch 20: Train Loss=0.1790, Val Loss=0.2884, Val mIoU=0.7163
```

最佳模型保存为 `best_unet_oxford_multiclass.pth`。

---

## 🔍 批量预测与交付物生成

编写了 `batch_predict_unet_with_deliverables.py`，实现以下功能：

1. **读取 `test_imgs` 文件夹中的任意图片**（支持 `.png`、`.jpg` 等）。
2. **对每张图片进行分割预测**，得到像素级掩码（0~36）。
3. **生成两类输出**（分文件夹存放）：

   - **可视化结果**（`test_results_paired/`）：
     - `*_paired.png`：原图（缩放后）与彩色分割掩码的左右拼合图。
     - `*_index.png`：原始类别索引灰度图（像素值为 0~36，用于精确验证）。

   - **交付数据**（`test_results_deliverables/`）：
     - `*_stats.json`：每张图片的统计信息，包含各类别像素数和占比。
     - `summary.csv`：所有图片的汇总表，每行一张图的像素统计。

**JSON 示例**：
```json
{
  "image_file": "cat_dog.jpg",
  "total_pixels": 65536,
  "class_pixel_counts": {"0": 50000, "1": 8000, "2": 7540},
  "class_ratios": {"0": 0.763, "1": 0.122, "2": 0.115}
}
```

**文件夹结构**：
```
项目目录/
├── test_imgs/                         # 输入图片（用户自行放入）
├── test_results_paired/                # 可视化结果
│   ├── img1_paired.png
│   ├── img1_index.png
│   └── ...
├── test_results_deliverables/           # 交付数据
│   ├── img1_stats.json
│   ├── ...
│   └── summary.csv
├── best_unet_oxford_multiclass.pth      # 训练好的模型权重
└── batch_predict_unet_with_deliverables.py
```

---

## 🧪 验证与调优

- **模型能否区分多动物图片？**  
  用一张同时包含猫和狗的图片测试，读取 `*_index.png` 的像素值，猫区域为 1，狗区域为 2，说明模型正确区分了不同类别（尽管彩色图中颜色接近）。

- **颜色映射优化**  
  从最初的 `jet` 到 HSV 均匀采样，最终采用 **`tab20` + `tab20b` 组合色板**，使 37 类的颜色区分度明显提高，便于肉眼观察。

- **交付数据格式**  
  考虑到数模组需要的是量化数据而非图像，因此将统计信息保存为 JSON 和 CSV，方便后续时序分析。

---

## 🚧 遇到的坑与解决方案

| 问题                                 | 现象                                                                 | 解决                                                                 |
| ------------------------------------ | -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| DLL 加载失败                         | `OSError: [WinError 1114] shm.dll`                                   | 重启 PyCharm/终端后消失，或降级 PyTorch 到 2.2.2                     |
| 通道数不匹配                         | `RuntimeError: expected input ... to have 1024 channels, but got 1536` | 修正 `Up` 模块的输入通道数为拼接后的总通道数（1024+512 等）           |
| 灰度索引图全黑                       | 保存的 `*_index.png` 看起来纯黑                                       | 正常现象，因为像素值 0~36 在 8 位灰度图中非常暗，可用代码读取具体数值 |
| 多动物图片颜色相近                   | 猫和狗在彩色图中颜色难以区分                                           | 改用 `tab20`+`tab20b` 色板，并保留索引图供数值验证                   |
| Matplotlib 弃用警告                   | `get_cmap` 已弃用                                                     | 使用 `matplotlib.colormaps` 或 `ListedColormap` 自定义颜色列表       |

---

## 🔜 迁移到藻类项目

本项目的代码结构和交付物格式可直接迁移到藻类识别任务中，只需修改：

1. **类别数**：`num_classes = 6`（水、蓝藻、绿藻、其他藻、泥沙、阴影）。
2. **模型权重**：加载在藻类数据集上训练好的 U‑Net 模型。
3. **数据预处理**：根据实际图像尺寸和归一化方式调整 `transform`。
4. **输出文件夹**：重命名为项目相关的名称（如 `algae_vis`、`algae_deliverables`）。
5. **类别名称映射**：可在 JSON/CSV 中添加字段说明类别含义。

---

## 📚 总结

通过本次实践，我：

- 深入理解了 U‑Net 的结构与工作原理。
- 掌握了从数据加载到模型训练、评估、预测的全流程。
- 学会了生成标准化的量化数据，为团队协作打下基础。
- 积累了解决环境配置、维度错误、颜色映射等实际问题的经验。

这些成果将直接支撑后续的无人机藻类智能识别项目。

---

**最后更新**：2026年3月18日  
**作者**：大创项目 AI 视觉模块成员 **吴天宇**