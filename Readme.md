# U-Net 语义分割学习实践记录 —— Oxford Pets 多类分割

> 本项目是我学习 U-Net 语义分割的练习记录，使用 Oxford-IIIT Pet 数据集（37 类）进行多类分割训练，并编写了批量预测脚本。目的是为后续的无人机藻类识别项目打下基础。

---

## 📌 项目概述

- **目标**：掌握 U-Net 的完整使用流程，包括数据加载、模型构建、训练、评估、预测。
- **数据集**：Oxford-IIIT Pet（37 类：背景 + 36 种宠物），通过 `torchvision` 自动下载。
- **框架**：PyTorch（已适配 RTX 5060 显卡 + Python 3.12）
- **核心成果**：
  - 成功训练 U-Net 模型，验证集 mIoU 达到 **0.72**。
  - 编写批量预测脚本，输入图片自动生成左右拼合图（原图 + 彩色分割掩码）。
  - 记录并解决了环境配置、模型维度错误等问题。

---

## 🛠️ 环境配置

使用 Conda 管理虚拟环境，关键依赖如下：

- Python 3.12
- PyTorch 2.2.2 + CUDA 11.8（实际使用 nightly 2.10.0 也兼容）
- torchvision 0.17.2
- matplotlib, numpy, tqdm, pillow

**创建环境命令**：
```bash
conda create --prefix E:\conda_env\unet_practice python=3.12 -y
conda activate E:\conda_env\unet_practice
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib tqdm pillow
```

---

## 📂 数据集准备

Oxford Pets 数据集由 `torchvision.datasets.OxfordIIITPet` 自动下载，无需手动处理。  
训练时对图像进行 **256×256 缩放**，掩码使用最近邻插值保持类别标签不变。

```python
# 关键代码
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64))),
])
```

数据集划分：80% 训练（2944 张），20% 验证（736 张）。

---

## 🧠 模型训练

### U-Net 结构
- 编码器：4 次下采样（64→128→256→512→1024）
- 解码器：4 次上采样 + 跳跃连接
- 输出通道：37（对应 37 个类别）

### 训练配置
- 损失函数：`nn.CrossEntropyLoss()`
- 优化器：Adam（lr=1e-3）
- 评估指标：mIoU（平均交并比）
- 训练轮数：20
- 批次大小：4（根据显卡显存调整）

### 训练日志（部分）
```
Epoch 1: Train Loss=0.8419, Val Loss=0.5381, Val mIoU=0.5420
Epoch 5: Train Loss=0.4095, Val Loss=0.3661, Val mIoU=0.6303
Epoch 10: Train Loss=0.3119, Val Loss=0.3283, Val mIoU=0.6709
Epoch 15: Train Loss=0.2429, Val Loss=0.3059, Val mIoU=0.7025
Epoch 20: Train Loss=0.1790, Val Loss=0.2884, Val mIoU=0.7163
```
- **最佳模型**：验证损失最低的 epoch（通常在第 13-16 轮），最终 mIoU ≈ **0.72**。

---

## 🖼️ 训练结果可视化

训练结束后，脚本自动保存了验证集 3 张图片的预测对比图（`oxford_pets_multiclass_prediction.png`）：

- 左：输入图像（缩放后）
- 中：真实掩码（彩色）
- 右：预测掩码（彩色）

> *（此处可插入图片预览，例如：![预测对比](oxford_pets_multiclass_prediction.png)）*

从图中可以看出，模型能够较好地分割出宠物轮廓，不同品种之间颜色区分明显。

---

## 🔍 批量预测脚本

为了方便测试新图片，编写了 `batch_predict_unet_paired.py`。它读取 `test_imgs` 文件夹中的图片，输出左右拼合图至 `test_results_paired` 文件夹。

### 使用方法
1. 将待测图片放入 `test_imgs` 目录。
2. 确保模型权重文件 `best_unet_oxford_multiclass.pth` 在脚本同级目录。
3. 运行脚本：
   ```bash
   python batch_predict_unet_paired.py
   ```
4. 在 `test_results_paired` 中查看 `原文件名_paired.png` 结果。

### 输出示例
![预测拼合示例](sample_paired.png)  
*（左：原图缩放至 256×256，右：模型预测的彩色分割掩码）*

---

## 🧩 文件结构

```
项目目录/
│
├── unet_oxford_pets_multiclass.py   # 训练脚本
├── batch_predict_unet_paired.py     # 批量预测脚本
├── best_unet_oxford_multiclass.pth  # 训练好的模型权重
├── oxford_pets_multiclass_prediction.png  # 训练结果可视化图
├── test_imgs/                        # 存放待测试图片（用户自己添加）
└── test_results_paired/               # 批量预测结果（自动生成）
```

---

## 🧪 遇到的坑与解决方案

| 问题 | 现象 | 解决 |
|------|------|------|
| DLL 加载失败 | `OSError: [WinError 1114] shm.dll` | 重启 PyCharm/终端后消失，或降级 PyTorch 到 2.2.2 |
| 通道数不匹配 | `RuntimeError: expected input[4, 1536, 32, 32] to have 1024 channels` | 修正 `Up` 模块的输入通道数为拼接后的总通道数 |
| 字体警告 | `Glyph ... missing from font` | 不影响结果，可忽略或设置中文字体 |
| 掩码像素值异常 | 发现掩码范围只有 1~3 | 确认数据集实际只包含部分类别，属于正常现象 |

---

## 🚀 下一步计划

1. **迁移到藻类数据集**：将 Oxford Pets 代码中的数据集类替换为自定义 `AlgaeDataset`，修改类别数为 6，并加载预处理后的藻类图像。
2. **处理类别不平衡**：针对藻类占比小的问题，引入加权损失或 Dice Loss。
3. **量化输出**：从预测掩码中计算藻类覆盖面积、占比，并生成结构化数据（JSON/CSV）供数模组使用。
4. **集成到展示端**：将模型预测结果对接可视化 Demo。

---

## 📚 总结

通过本次练习，我掌握了：
- U-Net 的完整实现与训练流程
- 多类分割的数据处理与评估方法
- 模型预测与结果可视化技巧

这为后续无人机藻类识别项目奠定了坚实基础。如果对本文档有任何疑问或建议，欢迎交流！

--- 
**最后更新**：2026年3月18日