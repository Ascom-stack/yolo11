# 基于边缘增强和双向特征融合的热轧钢带多尺度缺陷检测

[![论文](https://img.shields.io/badge/论文-The%20Visual%20Computer-blue)](YOUR_DOI_HERE)
[![代码](https://img.shields.io/badge/代码-GitHub-green)](YOUR_GITHUB_LINK)
[![数据集](https://img.shields.io/badge/数据集-NEU--DET-orange)](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)

> **重要提示**: 本代码库与提交至 **The Visual Computer** 期刊的论文直接相关。如果您使用了本代码或发现它对您的研究有帮助，请引用我们的论文：
> 
> ```bibtex
> @article{YOUR_CITATION_KEY,
>   title={基于边缘增强和双向特征融合的热轧钢带多尺度缺陷检测},
>   author={作者姓名},
>   journal={The Visual Computer},
>   year={2026},
>   doi={YOUR_DOI}
> }
> ```

## 摘要

热轧钢带是工业制造中的基础材料，其表面纹理复杂且易产生各种缺陷。传统的人工检测方法存在劳动强度大、主观性强等局限性。机器视觉方法虽然前景广阔，但在复杂表面的多尺度缺陷检测方面仍面临挑战。本文提出了一种基于边缘增强和双向特征融合的新型多尺度缺陷检测方法。设计了边缘信息（EI）模块，用于削弱背景纹理干扰并增强缺陷边缘特征提取。C3K2-DB模块结合了Fasterblock和EMA多尺度注意力机制，强化了缺陷类别属性提取。构建了RepBiPAN双向特征融合网络，整合跨尺度的缺陷特征信息，提高检测精度。在NEU-DET数据集上的实验结果表明，相比基线模型，本方法在精确率（83.9%）、召回率（84.4%）和mAP@0.5（85.6%）方面均有显著提升，为工业应用提供了有价值的参考。

## 主要特点

- **边缘信息增强（EIEStem）**: 削弱背景纹理干扰，增强缺陷边缘特征
- **C3K2-DB模块**: 结合Fasterblock和EMA多尺度注意力，改善缺陷类别提取
- **RepBiPAN网络**: 双向特征融合，实现多尺度缺陷检测
- **卓越性能**: 在NEU-DET数据集上达到83.9%精确率、84.4%召回率和85.6% mAP@0.5

## 模型架构

我们改进的模型（YOLO11n-ECR）包含：
- **主干网络**: EIEStem + C3K2_Faster_EMA模块
- **颈部网络**: BiFusion双向特征融合
- **检测头**: 基于RepBlock的检测头

模型配置文件：`ultralytics/cfg/models/11/yolo11n-ecr.yaml`

## 环境要求

### 环境配置

```bash
# 推荐使用Python 3.8+
pip install ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
pip install numpy
pip install matplotlib
pip install pyyaml
pip install tqdm
```

### 硬件要求

- **GPU**: 支持CUDA的NVIDIA显卡（推荐：RTX 3060或更高）
- **内存**: 最低16GB
- **存储**: 10GB用于数据集和模型权重

## 数据集准备

### NEU-DET数据集

NEU-DET数据集包含1,800张热轧钢带表面六种缺陷类型的灰度图像：
- 裂纹（Crazing, Cr）
- 夹杂（Inclusion, In）
- 麻点（Patches, Pa）
- 点蚀表面（Pitted Surface, PS）
- 压入氧化皮（Rolled-in Scale, RS）
- 划痕（Scratches, Sc）

### 下载和配置

1. 从[官方网站](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)下载NEU-DET数据集

2. 组织数据集结构：
```
NEU-DET/
├── images/
│   ├── train/          # 训练集图像
│   ├── val/            # 验证集图像
│   └── test/           # 测试集图像
└── labels/
    ├── train/          # 训练集标签
    ├── val/            # 验证集标签
    └── test/           # 测试集标签
```

3. 更新`dataset/you.yaml`中的数据集路径：
```yaml
path: /你的路径/NEU-DET
train:
  - images/train
  - labels/train
val:
  - images/val
  - labels/val
test:
  - images/test
  - labels/test

names:
  0: Crazing          # 裂纹
  1: Inclusion        # 夹杂
  2: Patches          # 麻点
  3: Pitted Surface   # 点蚀表面
  4: Rolled-in Scale  # 压入氧化皮
  5: Scratches        # 划痕
```

## 使用方法

### 训练模型

在NEU-DET数据集上训练YOLO11n-ECR模型：

```python
from ultralytics import YOLO

# 加载模型配置
model = YOLO('ultralytics/cfg/models/11/yolo11n-ecr.yaml')

# 训练模型
model.train(
    data='dataset/you.yaml',      # 数据集配置文件
    cache=False,                   # 不缓存图像
    imgsz=640,                     # 输入图像尺寸
    epochs=300,                    # 训练轮数
    batch=16,                      # 批次大小
    lr0=0.01,                      # 初始学习率
    close_mosaic=0,                # 关闭马赛克增强的轮数
    workers=0,                     # Windows系统建议设为0
    optimizer='SGD',               # 优化器
    patience=100,                  # 早停耐心值
    project='runs/train',          # 保存路径
    name='yolo11n-ecr',           # 实验名称
)
```

或者直接运行：
```bash
python train.py
```

**注意**: 运行前需要修改`train.py`中的`yaml_name`为`'yolo11n-ecr'`。

### 验证模型

评估训练好的模型：

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/yolo11n-ecr/weights/best.pt')

# 在测试集上验证
model.val(
    data='dataset/you.yaml',      # 数据集配置文件
    split='test',                  # 使用测试集
    imgsz=640,                     # 输入图像尺寸
    batch=16,                      # 批次大小
    project='runs/val',            # 保存路径
    name='yolo11n-ecr',           # 实验名称
)
```

或者运行：
```bash
python val.py
```

**注意**: 需要更新`val.py`中的模型权重路径。

### 推理预测

对新图像进行推理：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/train/yolo11n-ecr/weights/best.pt')

# 预测
results = model.predict(
    source='图像路径',              # 图像文件或文件夹路径
    imgsz=640,                     # 输入图像尺寸
    conf=0.25,                     # 置信度阈值
    save=True,                     # 保存结果
    project='runs/predict',        # 保存路径
    name='yolo11n-ecr'            # 实验名称
)
```

## 实验结果

### NEU-DET数据集性能

| 模型 | 精确率 (%) | 召回率 (%) | mAP@0.5 (%) | 参数量 (M) | FLOPs (G) |
|------|-----------|-----------|-------------|-----------|-----------|
| YOLOv11n (基线) | - | - | - | 2.6 | 6.6 |
| **YOLO11n-ECR (本文)** | **83.9** | **84.4** | **85.6** | - | - |

### 主要改进

- ✅ 通过EIEStem模块增强边缘特征提取
- ✅ 使用BiFusion改进多尺度特征融合
- ✅ 通过C3K2_Faster_EMA提升缺陷类别判别能力
- ✅ 对各种缺陷类型和尺度实现鲁棒检测

## 项目结构

```
.
├── ultralytics/
│   ├── cfg/
│   │   └── models/
│   │       └── 11/
│   │           ├── yolo11n-ecr.yaml      # 我们的改进模型
│   │           └── yolo11-EIEStem.yaml   # EIEStem变体
│   ├── nn/
│   │   ├── modules/                      # 自定义模块
│   │   └── tasks.py                      # 模型任务
│   └── ...
├── dataset/
│   ├── you.yaml                          # 数据集配置
│   └── ...
├── NEU-DET/                              # 数据集目录
│   ├── images/
│   └── labels/
├── runs/                                 # 训练/验证输出
│   ├── train/
│   ├── val/
│   └── test/
├── train.py                              # 训练脚本
├── val.py                                # 验证脚本
├── README.md                             # 英文说明文档
└── README_CN.md                          # 中文说明文档（本文件）
```

## 自定义模块

我们的实现包含以下自定义模块：

1. **EIEStem**: 边缘信息增强Stem，用于初始特征提取
2. **C3K2_Faster_EMA**: 结合Fasterblock和EMA注意力机制
3. **BiFusion**: 双向特征融合，实现多尺度整合
4. **RepBlock**: 重参数化模块，提高推理效率

这些模块已集成到Ultralytics框架中，可在`ultralytics/nn/modules/`中找到。

## 训练技巧

1. **显存不足**: 如果遇到OOM错误，请减小`batch`大小
2. **多进程问题**: Windows系统下设置`workers=0`避免多进程问题
3. **AMP问题**: 如果loss出现NaN，尝试设置`amp=False`关闭混合精度训练
4. **断点续训**: 使用`resume=True`并加载`last.pt`继续中断的训练
5. **早停控制**: 调整`patience`参数控制早停策略

## 常见问题

### 1. 如何指定GPU训练？

在`train.py`中添加`device`参数：
```python
model.train(
    device='0',  # 使用第一块GPU
    # 或 device='0,1' 使用多GPU
    ...
)
```

### 2. 如何修改类别数量？

修改`yolo11n-ecr.yaml`中的`nc`参数：
```yaml
nc: 6  # 修改为你的类别数量
```

### 3. 如何保存COCO格式的结果？

在`val.py`中添加：
```python
model.val(
    save_json=True,  # 保存COCO格式结果
    ...
)
```

### 4. 训练时显存占用过高怎么办？

- 减小batch size
- 减小图像尺寸（imgsz）
- 使用更小的模型变体

### 5. 如何可视化训练过程？

训练结果会自动保存在`runs/train/实验名称/`目录下，包括：
- `results.png`: 训练曲线
- `confusion_matrix.png`: 混淆矩阵
- `val_batch*.jpg`: 验证批次可视化

## 引用

如果您使用了本代码或发现我们的工作有帮助，请引用我们的论文：

```bibtex
@article{YOUR_CITATION_KEY,
  title={基于边缘增强和双向特征融合的热轧钢带多尺度缺陷检测},
  author={作者姓名},
  journal={The Visual Computer},
  year={2026},
  doi={YOUR_DOI}
}
```

## 许可证

本项目基于AGPL-3.0许可证发布。详见[LICENSE](LICENSE)。

## 致谢

- 本工作基于[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- NEU-DET数据集由东北大学提供
- 特别感谢开源社区的贡献

## 联系方式

如有问题或建议，请：
- 在GitHub上提交Issue
- 联系邮箱：[您的邮箱]

## 相关资源

- [Ultralytics官方文档](https://docs.ultralytics.com/)
- [NEU-DET数据集](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)
- [The Visual Computer期刊](https://www.springer.com/journal/371)
- [YOLO11官方仓库](https://github.com/ultralytics/ultralytics)

## 更新日志

- **2026-02**: 初始版本发布
- 提供完整的训练和验证代码
- 包含NEU-DET数据集配置
- 提供详细的使用文档

---

**可复现性声明**: 本仓库提供了复现论文实验结果所需的全部代码、配置和说明。我们致力于开放科学和透明的研究实践。

**DOI分配**: 为确保代码和数据的长期可访问性和可引用性，我们建议：
- 在GitHub上发布代码并创建Release版本
- 通过Zenodo为代码分配DOI
- 在论文中明确标注代码和数据的永久链接

**论文相关性提醒**: 本代码库直接对应提交至The Visual Computer期刊的研究论文。使用本代码进行研究时，请务必引用相关论文。
