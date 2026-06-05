# Dual-Attention RegNet for Wheat Disease Recognition

>官方 PyTorch 实现 | 论文处于已录用阶段，标题：《Dual-Attention RegNet for Wheat Disease Recognition》

>提出基于改进 RegNet 架构，融合可变形注意力（DAT）与图注意力（GAT）的小麦叶片病害识别模型，实现健康、叶锈病、白粉病、壳针孢叶枯病的高精度分类，辅助智慧农业中的精准植保。

## 1. 研究背景与模型定位

小麦叶部病害（如叶锈病、白粉病、壳针孢叶枯病）易导致产量损失 10%~30%。传统人工检测效率低、主观性强。现有深度学习模型在田间应用中存在延迟高、误报率高等问题。

本文提出基于改进 RegNet 与双注意力融合的小麦病害识别模型：

- 扩大池化层的输入单元和输出维度，增强多尺度空间特征捕获；
- 在浅层和深层嵌入**可变形注意力（DAT）**，自适应聚焦不规则病斑区域；
- 在 Stage 3 嵌入**图注意力网络（GAT）**，通过 3×3 邻域构建局部图结构，建模病斑间的空间依赖关系。

在自建小麦病害数据集上达到 **97.72%** 的总体分类准确率，适合边缘部署，为智慧农业中的作物健康监测提供轻量化技术方案。
## 2. 核心创新点

- **扩大池化层感受野**：在不改变核心网络结构的前提下扩大全局平均池化层的输入单元和输出维度，增强多尺度病斑特征的聚合能力，同时降低计算量，适合资源受限的边缘设备。
- **可变形注意力（DAT）**：嵌入 RegNet 的浅层（Stage 1）和深层（Stage 3）。浅层 DAT 增强病斑局部几何结构与边缘线索，深层 DAT 强化病斑与背景的语义区分度。通过动态偏移学习使采样点自适应聚焦于不规则病斑（如锈病孢子堆的随机散布），减少漏检和误检。
- **图注意力网络（GAT）**：仅嵌入 RegNet 的 Stage 3（中层特征）。将特征图节点按 3×3 邻域构建图结构，利用多头注意力建模病斑区域间的空间相关性，捕捉病斑蔓延的生物学过程，提升小目标病害（如早期白粉病）的检测精度。
- **轻量化设计**：通过模块简化与通道压缩，最终模型参数量仅 **2.65M**，在保证高精度的同时满足无人机/手持终端的实时推理需求。
  
## 3. 实验数据集：Wheat Disease Dataset

### 3.1 数据集概况

数据集来源于小麦种植基地，包含原始图像（141 张）和通过 GAN 生成的增强图像（5,699 张），共计 **5,840 张**。覆盖四个类别：健康、叶锈病、白粉病、壳针孢叶枯病。划分比例为训练:验证:测试 = 3:1:1。

| 类别 | 训练集 | 验证集 | 测试集 | 总计 |
|------|--------|--------|--------|------|
| 健康 | 951 | 342 | 342 | 1635 |
| 叶锈病 | 783 | 261 | 261 | 1305 |
| 白粉病 | 912 | 304 | 304 | 1520 |
| 壳针孢叶枯病 | 828 | 276 | 276 | 1380 |

### 3.2 数据集结构

请将图像按以下目录结构放置（图像为 RGB 格式，统一 resize 至 224×224）：

```text
Wheat3/
├── train/
│ ├── healthy/
│ ├── leaf_rust/
│ ├── powdery_mildew/
│ └── septoria/
├── val/
│ ├── healthy/
│ ├── leaf_rust/
│ ├── powdery_mildew/
│ └── septoria/
└── test/
├── healthy/
├── leaf_rust/
├── powdery_mildew/
└── septoria/
```
通过网盘分享的文件：wheat3.zip
链接: https://pan.baidu.com/s/1hdrZkE19nNrg9tRL0ZjJhA
## 4. 实验环境配置

### 4.1 依赖安装

推荐使用 Anaconda 创建虚拟环境，Python 3.10，PyTorch 2.0.1（已验证兼容）：

```bash
# 1. 创建并激活虚拟环境
conda create -n regnet-dual python=3.10
conda activate regnet-dual

# 2. 安装 PyTorch（CUDA 11.8 示例，CPU 用户可替换为 cpu 版本）
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖库
pip install timm==1.0.15 einops==0.8.1 tensorboard==2.10.0 tqdm pillow numpy
```

### 4.2 硬件要求
GPU：推荐 NVIDIA GPU（显存 ≥ 6GB，如 RTX 3060/4060），训练 100 轮约 2-3 小时，显存占用峰值 ≤ 5GB。

CPU：支持推理测试（单张图像约 15 ms），但不推荐用于完整训练流程。

## 5. 实验结果

### 5.1 核心指标对比（Academic Dataset）
本文方法（DGRNet = Improved RegNet + DAT + GAT）与多种主流模型在英语学业达标预测任务上的性能对比如下：

```bash
本文方法（DGRNet）在小麦病害数据集上取得了 97.68% 的总体准确率，参数量仅 2.65M，在 RTX 4060 上单张推理时间小于 20 毫秒。

消融实验表明：单独引入可变形注意力（DAT）后，准确率从基线的 92.51% 提升至 95.52%；单独引入图注意力（GAT）后，准确率提升至 94.35%；二者联合使用时，准确率达到 97.68%，同时灵敏度、精确率、特异性和 F1 分数均优于单一模块，验证了双注意力协同的有效性。

各类别识别精度：健康叶片 > 97.2%，叶锈病 > 83%，白粉病 > 95.5%，壳针孢叶枯病 > 85%。
```

## 6. 代码使用说明

### 6.1 模型训练
模型定义位于 model3.py，核心类为 RegNetWithAttention。为匹配小麦病害识别任务（4 类），需要正确配置 dattention_config 和 gat_config，且必须启用它们，否则模型退化为纯 RegNet。

```bash
from model3 import create_regnet_with_attention

dattention_config = {
    'use_dattention': True,
    'stages': [1, 3],          # 论文中浅层(stage1)和深层(stage3)嵌入DAT
    'n_heads': 8,
    'n_groups': 4,
}
gat_config = {
    'use_gat': True,
    'stages': [3],             # 仅stage3嵌入GAT
    'nhid': 64,
    'nheads': 4,
    'dropout': 0.1,
    'alpha': 0.2,
}

model = create_regnet_with_attention(
    model_name='regnety_400mf',
    num_classes=4,                     # 小麦病害 4 类
    dattention_config=dattention_config,
    gat_config=gat_config
)
```
### 6.2 模型预测
训练脚本 train3.py 已适配小麦数据集（num_classes=4, data-path 指向你的数据集根目录）。
注意：默认配置未启用 DAT/GAT，请务必在创建模型时传入上述配置。推荐超参数（源自论文 Table 2 最佳配置）：

```bash
python train3.py \
  --data-path ./WheatDataset \
  --model-name regnety_400mf \
  --num-classes 4 \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.0002 \
  --device cuda:0
```

### 6.3 模型评估与预测
评估示例：
```bash
import torch
from model3 import create_regnet_with_attention
from utils import evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_regnet_with_attention(
    "regnety_400mf", num_classes=4,
    dattention_config={...},    # 与训练时一致
    gat_config={...}
)
model.load_state_dict(torch.load("./weights3/best_model.pth"))
model.to(device)

test_acc = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}")
```

## 7.项目文件结构
```bash
wheat-disease-recognition/
├── WheatDataset/                # 小麦病害数据集（按类别分文件夹）
│   ├── train/
│   ├── val/
│   └── test/
├── weights3/                    # 训练保存的模型权重
├── model3.py                    # 模型定义（含改进 RegNet + DAT + GAT）
├── train3.py                    # 训练脚本
├── my_dataset.py                # 自定义 DataLoader
├── utils.py                     # 评估、训练辅助函数
├── training3.txt                # 训练日志
└── README.md                    # 本文档
```
## 8. 已知问题与注意事项
注意力配置必须显式启用：若 train3.py 中未传递 dattention_config 和 gat_config，模型实际为纯 RegNet，性能将远低于论文报告值。

数据集局限：原始图像仅 141 张，其余为 GAN 生成，可能存在合成伪影。模型在真实田间环境下的泛化能力需额外验证。

病害类别不全：当前仅覆盖健康、叶锈病、白粉病、壳针孢叶枯病，缺少赤霉病、纹枯病等常见病害。

环境适应性：现有数据多采集于特定生长季节和天气条件（阴天散射光），强光、晨露、不同生育期下的性能需额外测试。

显存与批次：若 batch size = 64 需约 6GB 显存，可适当降低至 32 或使用梯度累积。

## 9. 引用与联系方式
### 9.1 引用方式
论文已发表于 Information 期刊，请使用以下 BibTeX 格式引用：
```bash
@article{li2025dual,
  title={Dual-Attention RegNet for Wheat Disease Recognition},
  author={Li, Gang and Wang, Xiaowei and Zhao, Hao and Xu, Peng and Du, Xiaojie and Xu, Laixiang},
  journal={（待补充）},
  year={2025},
  note={Submitted for publication}
}
```
### 9.2 联系方式
若遇到代码运行问题或学术交流需求，请联系：
邮箱：wangxiaowei@huuc.edu.cn

GitHub Issue：直接在本仓库提交 Issue，会在 1-3 个工作日内回复。
