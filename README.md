# 数字图像处理（Digital Image Processing）

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/Tests-281%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个完整的数字图像处理算法库，包含71+种经典算法的Python实现，适用于教学、研究和实践应用。

## 📊 项目概览

- **11个功能模块** - 系统化的算法组织
- **71+种算法** - 从基础到高级的完整实现
- **281个单元测试** - 全部通过，代码质量保证
- **15,000+行代码** - 清晰的实现和注释
- **完整中文文档** - 详细的算法说明和使用指南

## 📁 项目结构

```
DIP/
├── config/                 # 配置文件
├── docs/                   # 文档（算法原理、使用指南）
├── src/                    # 源代码
│   ├── frequency_filtering/          # 频域滤波
│   ├── image_operations/             # 图像运算
│   ├── spatial_filtering/            # 空间域滤波
│   ├── noise_processing/             # 噪声处理
│   ├── mean_filters/                 # 均值滤波器
│   ├── order_statistic_filters/      # 统计排序滤波器
│   ├── color_image_processing/       # 彩色图像处理
│   ├── image_compression/            # 图像压缩编码
│   ├── morphological_operations/     # 形态学处理
│   ├── edge_detection/               # 边缘检测
│   └── advanced_segmentation/        # 高级图像分割
├── tests/                  # 单元测试
├── examples/               # 示例图像和结果
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用示例

```python
# 示例1: Canny边缘检测
import sys
sys.path.append('src')
from advanced_segmentation.advanced_segmentation import AdvancedSegmentation
import cv2

seg = AdvancedSegmentation()
image = cv2.imread('examples/test_image.png', cv2.IMREAD_GRAYSCALE)
edges, steps = seg.canny_edge_detection(image, 50, 150)
cv2.imwrite('output.png', edges)
```

```python
# 示例2: 中值滤波去除椒盐噪声
from order_statistic_filters.order_statistic_filters import OrderStatisticFilters

filters = OrderStatisticFilters()
noisy = cv2.imread('examples/noisy.png', cv2.IMREAD_GRAYSCALE)
denoised = filters.median_filter(noisy, kernel_size=5)
```

```python
# 示例3: OTSU自动阈值分割
from advanced_segmentation.advanced_segmentation import AdvancedSegmentation

seg = AdvancedSegmentation()
threshold, binary = seg.otsu_threshold(image)
print(f"最优阈值: {threshold}")
```

## 🎯 功能模块

### 1. 频域滤波 (`src/frequency_filtering/`)
- 低通滤波器（理想、巴特沃斯、高斯）
- 陷波滤波器（去除周期性噪声）
- 带通/带阻滤波器

### 2. 图像运算 (`src/image_operations/`)
- 算术运算（加、减、乘、除）
- 逻辑运算（AND、OR、NOT、XOR）

### 3. 空间域滤波 (`src/spatial_filtering/`)
- 均值滤波、中值滤波
- 拉普拉斯锐化、高通滤波

### 4. 噪声处理 (`src/noise_processing/`)
- 12种噪声模拟（高斯、椒盐、泊松等）
- 噪声识别与估计

### 5. 均值滤波器 (`src/mean_filters/`)
- 算术、几何、谐波、逆谐波均值

### 6. 统计排序滤波器 (`src/order_statistic_filters/`)
- 中值、最大值、最小值、中点、修正阿尔法均值

### 7. 彩色图像处理 (`src/color_image_processing/`)
- RGB/HSV/LAB颜色空间
- 平滑与锐化（双边滤波、Unsharp Masking）

### 8. 图像压缩编码 (`src/image_compression/`)
- 哈夫曼、算术、LZW、位平面编码

### 9. 形态学处理 (`src/morphological_operations/`)
- 腐蚀、膨胀、开运算、闭运算
- 形态学梯度、顶帽、黑帽

### 10. 边缘检测 (`src/edge_detection/`)
- Roberts、Prewitt、Sobel、Canny
- 点检测、线检测

### 11. 高级图像分割 (`src/advanced_segmentation/`)
- Canny详细实现（5步骤）
- Hough变换（直线/圆检测）
- OTSU自动阈值、分水岭分割

详细功能列表请查看 [FEATURES_LIST.md](FEATURES_LIST.md)

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_edge_detection.py -v

# 查看测试覆盖率
pytest --cov=src tests/
```

**测试统计**: 281个单元测试，全部通过 ✅

## 📚 核心算法

### ⭐⭐⭐ 工业标准
- **Canny边缘检测** - 最优边缘检测
- **Sobel算子** - 最常用的边缘检测
- **中值滤波** - 椒盐噪声去除最优
- **OTSU阈值** - 自动阈值分割
- **高斯滤波** - 标准图像平滑
- **FFT变换** - 频域处理
- **形态学操作** - 二值图像处理

### ⭐⭐ 常用算法
- Hough变换、双边滤波、Unsharp Masking
- 分水岭分割、哈夫曼编码

### ⭐ 特定应用
- Roberts/Prewitt算子、统计排序滤波器
- 噪声模拟、LZW编码、陷波滤波器

## 🎓 学习路径

**入门** → 图像运算 → 空间域滤波 → 边缘检测

**中级** → 噪声处理 → 均值滤波 → 形态学

**高级** → 频域滤波 → 高级分割 → 图像压缩

**专家** → 彩色处理 → 统计排序滤波

## 🏭 应用领域

| 领域 | 应用模块 |
|------|---------|
| **医学影像** | 噪声处理、边缘检测、分割 |
| **工业检测** | 边缘检测、Hough变换、形态学 |
| **文档处理** | 形态学、阈值分割、线检测 |
| **照片处理** | 彩色处理、锐化、去噪 |

## 📖 文档

- [完整功能清单](FEATURES_LIST.md) - 71+种算法详解
- [docs/](docs/) - 各模块的详细文档

## 🛠️ 技术特色

- **完整性** - 从基础到高级的完整算法体系
- **工程化** - 模块化设计，统一API接口
- **教学友好** - 详细注释，原理说明
- **实用性** - 工业级实现，基于OpenCV优化

## 📊 项目统计

```
模块数量      11 个
算法实现      71+ 种
代码行数      15,000+ 行
单元测试      281 个
示例图片      100+ 张
```

## 🤝 贡献

欢迎贡献！请提交Pull Request。

## 📄 许可证

MIT License

## 👥 作者

- **kiwios-cn** - 原始实现
- **Claude Opus 4.7** - AI辅助

---

⭐ **如果这个项目对你有帮助，请给一个Star！**
