# HRNet Facial Landmark Detection for WFLW

本项目基于 HRNet 人脸关键点检测方法进行整理与二次开发，主要用于 **WFLW 数据集上的人脸关键点检测训练、测试与 ONNX 推理部署**。当前版本主要完成了 WFLW 数据集下的模型训练流程、测试流程以及部分 ONNX 导出和单张图像推理脚本整理。

> 说明：本仓库代码参考并改写自 HRNet-Facial-Landmark-Detection 官方实现，当前主要用于学习、实验复现和后续工程化开发。

## 1. 项目简介

人脸关键点检测是人脸分析任务中的基础环节，可用于人脸对齐、表情分析、眼部区域定位、眨眼识别、人机交互等应用场景。本项目采用 HRNet 作为主干网络，通过保持高分辨率特征表示，实现对人脸关键点位置的精细预测。

当前项目重点关注：

- 基于 HRNet 的人脸关键点检测；
- WFLW 数据集训练与测试；
- 模型权重加载与结果评估；
- ONNX 模型导出；
- 单张图像 ONNX 推理测试；

## 2. 当前支持情况

目前本项目仅在 **WFLW 数据集** 上进行了训练和测试。

| 数据集 | 关键点数量 | 当前支持状态 |
|---|---:|---|
| WFLW | 98 | 已支持训练与测试 |
| 300W | 68 | 暂未验证 |
| AFLW | 19 | 暂未验证 |
| COFW | 29 | 暂未验证 |

后续如果补充其他数据集的训练结果，会继续更新说明。

## 3. 项目结构

```bash
HRNet/
├── experiments/                  # 配置文件
│   └── wflw/
│       └── face_alignment_wflw_hrnet_w18.yaml
├── lib/
│   ├── config/                   # 配置加载
│   ├── core/                     # 训练、测试和评估逻辑
│   ├── datasets/                 # 数据集读取
│   ├── models/                   # HRNet 网络结构
│   └── utils/                    # 工具函数
├── tools/
│   ├── train.py                  # 训练脚本
│   ├── test.py                   # 测试脚本
│   ├── export_hrnet_heatmap_coords_onnx.py   # ONNX 导出脚本
│   └── test_hrnet_onnx_image.py              # ONNX 单图推理脚本
├── data/                         # 数据集目录，需自行准备
├── output/                       # 训练输出目录
├── log/                          # 日志目录
└── README.md
```

## 4. 环境配置

建议使用 Conda 创建独立环境：

```bash
conda create -n hrnet-face python=3.8 -y
conda activate hrnet-face
```

安装项目运行所需依赖：

```bash
pip install torch torchvision
pip install opencv-python numpy scipy yacs tqdm matplotlib
pip install onnx onnxruntime
```
---

## 5. 数据集准备

当前项目仅在 **WFLW 数据集** 上进行了训练和测试。请自行下载 WFLW 数据集，并按照配置文件中的路径进行组织。注释文件和预训练权重可以在这里下载(
链接: https://pan.baidu.com/s/1OmxKxjGB1XJnwtltLHx7Lg 提取码: enq5)

推荐目录结构如下：

````
HRNet-Facial-Landmark-Detection
-- lib
-- experiments
-- hrnetv2_pretrained
   |--hrnetv2_w18_imagenet_pretrained.pth
-- tools
-- data
   |-- wflw
   |   |-- face_landmarks_wflw_test.csv
   |   |-- face_landmarks_wflw_test_blur.csv
   |   |-- face_landmarks_wflw_test_expression.csv
   |   |-- face_landmarks_wflw_test_illumination.csv
   |   |-- face_landmarks_wflw_test_largepose.csv
   |   |-- face_landmarks_wflw_test_makeup.csv
   |   |-- face_landmarks_wflw_test_occlusion.csv
   |   |-- face_landmarks_wflw_train.csv
   |   |-- images

````

对应配置文件路径为：

```text
experiments/wflw/face_alignment_wflw_hrnet_w18.yaml
```

需要重点检查以下字段，并根据实际数据路径进行修改：

```yaml
DATASET:
  ROOT: './data/wflw/images/'
  TRAINSET: './data/wflw/face_landmarks_wflw_train.csv'
  TESTSET: './data/wflw/face_landmarks_wflw_test.csv'
```

---

## 6. 模型训练

使用以下命令在 WFLW 数据集上训练 HRNet-W18 模型：

```bash
python tools/train.py --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml
```

训练完成后，模型权重默认保存在：

```text
output/WFLW/face_alignment_wflw_hrnet_w18/
```

---

## 7. 模型测试

使用训练好的模型进行测试：

```bash
python tools/test.py \
  --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml \
  --model-file output/WFLW/face_alignment_wflw_hrnet_w18/model_best.pth
```

如果模型权重路径不同，请根据实际保存位置修改 `--model-file` 参数。

---

## 8. ONNX 模型导出

本项目整理了 HRNet 关键点检测模型的 ONNX 导出脚本，可使用以下命令导出模型：

```bash
python tools/export_hrnet_heatmap_coords_onnx.py \
  --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml \
  --model-file output/WFLW/face_alignment_wflw_hrnet_w18/model_best.pth \
  --output hrnet_wflw.onnx
```

导出的 ONNX 模型可用于后续部署、推理测试或工程化集成。

---

## 9. ONNX 单张图像推理

使用 ONNX Runtime 对单张图像进行推理测试：

```bash
python tools/test_hrnet_onnx_image.py \
  --onnx hrnet_wflw.onnx \
  --image path/to/test.jpg
```

该脚本主要用于验证 ONNX 模型是否能够正常加载，并输出人脸关键点预测结果。

---

## 10. 预训练模型说明

如果使用 ImageNet 预训练的 HRNet 权重，请将其放置到对应目录，例如：

```text
hrnetv2_pretrained/
└── hrnetv2_w18_imagenet_pretrained.pth
```

由于预训练权重文件通常较大，建议不要直接上传到 GitHub 仓库中。可以通过网盘、GitHub Release 或 Git LFS 等方式管理模型权重。

---

## 11. 当前版本说明

当前版本主要完成以下内容：

- 整理 WFLW 数据集训练配置；
- 完成 HRNet-W18 在 WFLW 数据集上的训练与测试流程；
- 新增 ONNX 导出脚本；
- 新增 ONNX 单张图像推理测试脚本；
- 为后续眼部区域定位、眨眼识别和人机交互应用提供基础。

目前项目仅在 **WFLW 数据集** 上进行了训练和测试，其他数据集暂未进行系统验证。

---

## 12. 后续计划

- [ ] 补充 WFLW 数据集上的训练结果和测试指标；
- [ ] 增加关键点检测结果可视化示例；
- [ ] 整理模型权重下载方式；
- [ ] 支持摄像头实时人脸关键点检测；
- [ ] 与眼部区域检测、眨眼分类模块进行集成；
- [ ] 补充技术报告和数据集构建说明；
- [ ] 进一步优化模型推理速度和工程部署流程。

---

## 13. 致谢

本项目参考了 HRNet-Facial-Landmark-Detection 的代码实现，在此对原作者的工作表示感谢。HRNet 在保持高分辨率特征表达方面具有较好的关键点定位能力，为本项目在 WFLW 数据集上的训练、测试和后续应用开发提供了重要基础。

同时，本项目当前主要面向 WFLW 数据集下的人脸关键点检测实验与工程化整理，后续将结合实际应用需求继续完善。
