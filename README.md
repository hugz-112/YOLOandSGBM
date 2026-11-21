# YOLOv8 + ZED2 双目测距项目

本项目基于 YOLOv8（内置 `ultralytics 8.2.18`）实现目标检测、训练、验证、跟踪与区域计数，并结合双目视觉完成相机标定、极线矫正、深度图计算与双目测距（BM / SGBM / 改进 SGBM），同时提供 ZED2 相机 + YOLOv8 的 3D 目标检测示例。

## 项目简介

本项目基于 YOLOv8 和 SGBM，实现了水下目标检测与双目测距。

### 功能特点

- 支持实时目标检测；
- 支持双目测距；
- 支持 PyQt5 图形界面；
- 支持 TensorRT 加速部署。

> 注意：请先下载模型权重，再运行程序。

## 目录结构

```
YOLOv8.2/
├── train.py                     # 模型训练脚本
├── val.py                       # 模型验证脚本
├── track.py                     # 目标跟踪脚本
├── VideoHeat.py                 # 视频热力图生成
├── yolov8_region_counter.py     # 多边形区域人数/目标计数
├── ceju/                        # 双目视觉测距相关代码
│   ├── 1_BiaoDing/              # 相机标定与标定验证、Q 矩阵计算
│   ├── 2_JiaoZhen/              # 极线矫正（含左右样例图）
│   ├── 3_ShenDu/                # 多算法深度图对比评估
│   ├── 4_BM/                    # BM 双目测距
│   ├── 5_SGBM/                  # SGBM 实时视频测距
│   ├── 6_GaiJinSGBM/            # 改进 SGBM（UCT-SGM / 消融实验）
│   ├── 8_ZED2_pytorch_yolov8/   # ZED2 + YOLOv8 3D 检测跟踪示例
│   ├── advanced_augmentation.py # 数据增强工具（GUI）
│   ├── mosaic.py                # Mosaic 增强工具（GUI）
│   └── jihuohanshu.py           # 常用激活函数曲线可视化
└── ultralytics/                 # 内置 ultralytics 源码（含模型配置）
```

## 环境要求

- 实测环境：Python 3.10、PyTorch 2.1、CUDA 12.1
- Python 3.8+
- PyTorch / torchvision（GPU 训练需 CUDA）
- opencv-python、numpy、shapely、matplotlib、scikit-learn、Pillow
- pyzed + ZED SDK（仅 `8_ZED2_pytorch_yolov8` 需要）
- tkinter（Python 标准库，数据增强 GUI 使用）

```bash
pip install torch torchvision opencv-python numpy shapely matplotlib scikit-learn pillow
```

## 快速开始

### 1. 训练

编辑 [train.py](train.py) 中的数据集路径与超参数后运行：

```bash
python train.py
```

默认配置：`yolov8n.yaml` 从零训练，`imgsz=640`、`epochs=300`、`batch=8`、`optimizer=SGD`、`device='0'`，结果保存在 `runs/biye/shacnhu/`。

- 如需加载预训练权重，取消 `model.load('yolov8n.pt')` 的注释。
- 如需续训，设置 `resume` 为 `last.pt` 的路径。
- 切换模型规模时修改 `ultralytics/cfg/models/v8/yolov8n.yaml` 中的型号（如 `yolov8s.yaml`）。

### 2. 验证

修改 [val.py](val.py) 中的权重与数据路径后运行：

```bash
python val.py
```

默认使用 `runs/DUO.py3.10.cuda12.1/yolov8n-MLCA/weights/best.pt`，在 `MyDataset/DUO-main-4/data.yaml` 的验证集上评估，结果保存在 `runs/val/exp/`。

### 3. 目标跟踪

```bash
python track.py
```

使用 `yolov8n.pt` 对 `People.mp4` 进行跟踪，结果保存在 `runs/track/exp/`。

### 4. 热力图

```bash
python VideoHeat.py
```

对 `People.mp4` 逐帧跟踪并累计热力图，输出 `heatmap_output.avi`，同时实时窗口预览。

### 5. 多边形区域计数

```bash
python yolov8_region_counter.py --source People.mp4 --weights yolov8n.pt --device 0
```

基于 ByteTrack 统计落在多边形/矩形区域内的目标数量，运行时可鼠标拖动区域，结果保存在 `ultralytics_rc_output/exp/`。主要参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--weights` | 模型权重 | `yolov8n.pt` |
| `--source` | 视频路径 | `People.mp4` |
| `--device` | `0` 表示 CUDA，`cpu` 表示 CPU | `0` |
| `--classes` | 只统计指定类别 | `0` |
| `--view-img` / `--save-img` | 显示 / 保存结果 | `True` |

## 数据集格式

使用标准 YOLO 格式，`data.yaml` 需包含训练/验证图片目录与类别名称，例如：

```yaml
path: ../datasets/your_dataset
train: images/train
val: images/val
names:
  0: person
  1: car
```

> 仓库未包含 `MyDataset`、`runs`、`yolov8n.pt`、`People.mp4` 等数据与权重文件，请按自己的环境准备并修改脚本中的硬编码路径。

## 双目视觉模块（ceju）

### 1_BiaoDing — 相机标定

- `camera_calibration.py`：基于棋盘格（默认方格边长 0.03m、内角点 9×6）进行单目标定与双目标定，输出内参、畸变系数与标定 JSON。
- `calibration_verification.py`：加载标定结果，计算重投影误差，验证标定精度。
- `Q.py`：由左右相机内参与基线构造 4×4 重投影矩阵 Q（结果参考 `q_matrix.npy`）。

### 2_JiaoZhen — 极线矫正

- `epipolar_rectification.py`：读取 `left_19.jpg` / `right_19.jpg`，完成去畸变 + 立体校正并拼接对比图。
- `simplified_epipolar_rectification.py`：以类的方式封装极线矫正流程，便于复用。

### 3_ShenDu — 深度图对比

```bash
python ceju/3_ShenDu/depth_map_comparison.py
```

加载多张深度图，统一归一化与伪彩色显示，并用 MAE / RMSE 等指标定量对比不同算法的结果。

### 4_BM — BM 双目测距

`ceju/4_BM/BM.py` 基于 Block Matching 算法，面向 1920×1080 图像，内置鼠标点击测距功能。

### 5_SGBM — SGBM 实时测距

`ceju/5_SGBM/sgbm-video.py` 以 640×480 分辨率采集双目视频，实时计算视差/深度，鼠标左键点击可读取该点距离。

### 6_GaiJinSGBM — 改进 SGBM

`UCT_SGM_ablation.py` 实现 StarBoost-V3 UCT-SGM，主要改进：

- CLAHE + 双边滤波预处理，增强水下低对比度场景；
- AD-Census 代价构造，提升弱纹理区域匹配稳定性；
- 纹理阈值生成 ROI，仅保留手臂/海星/墙壁等结构区域视差；
- speckle 连通域去噪 + 形态学处理 + 连通域填充，修复空洞；
- 墙壁等结构边缘保留原始 SGM 视差，并叠加黄–蓝渐变配色。

`UCT_SGM_ablationv1.py` 为对应消融版本，便于对比各组件的贡献。

### 8_ZED2_pytorch_yolov8 — ZED2 + YOLOv8

将 YOLOv8 检测结果注入 ZED SDK，实现 3D 包围盒与目标跟踪：

```bash
python ceju/8_ZED2_pytorch_yolov8/detector.py --weights yolov8m.pt
```

需要安装 ZED SDK 与 pyZED，详细说明见该目录下的 [README](ceju/8_ZED2_pytorch_yolov8/README.md)。

### 数据增强工具

- `advanced_augmentation.py`：GUI 工具，支持色彩空间、旋转、缩放、透视、翻转、通道交换、Mosaic、Mixup、Cutmix 等增强的预览与保存。
- `mosaic.py`：GUI 工具，实现随机缩放、翻转、颜色扰动的 Mosaic 拼接增强，并对比预览、保存结果。

## 输出说明

| 脚本 | 输出目录 |
| --- | --- |
| `train.py` | `runs/biye/shacnhu/` |
| `val.py` | `runs/val/exp/` |
| `track.py` | `runs/track/exp/` |
| `yolov8_region_counter.py` | `ultralytics_rc_output/exp/` |
| `VideoHeat.py` | `heatmap_output.avi` |

## 注意事项

- 各脚本中的数据集、权重、视频路径均为硬编码示例，运行前请改为本机实际路径。
- `device='0'` 需要可用的 NVIDIA GPU；无 GPU 时改为 `'cpu'`。
- 训练出现 `NaN` 损失时可关闭 `amp`。
- 双目测距精度依赖标定结果，更换相机或调整基线后需重新标定并更新脚本中的内参、畸变与 R/T 参数。
