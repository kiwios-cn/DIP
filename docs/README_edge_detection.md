# 边缘检测

## 项目简介

本项目实现了完整的边缘检测算法，包括点检测、线检测和多种边缘检测算子。

### 点检测与线检测
- **点检测** - 检测孤立点
- **线检测** - 检测特定方向的线条（水平、垂直、对角线）

### 边缘检测算子
- **Roberts算子** - 最简单的边缘检测，2×2算子
- **Prewitt算子** - 3×3算子，简单平滑
- **Sobel算子** - 3×3算子，高斯加权，最常用
- **Canny算子** - 最优边缘检测算法

## 功能特点

1. ✅ **6种检测方法**：点、线、4种边缘算子
2. ✅ **梯度分析**：X/Y方向梯度和幅值
3. ✅ **完整测试**：25个单元测试
4. ✅ **可视化对比**：直观展示不同算子效果

## 快速开始

```python
from edge_detection import EdgeDetection
import cv2

detector = EdgeDetection()
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# 点检测
points = detector.point_detection(image, threshold=0.5)

# 线检测
h_lines = detector.line_detection(image, 'horizontal', threshold=0.5)
v_lines = detector.line_detection(image, 'vertical', threshold=0.5)

# 边缘检测
roberts_mag, gx, gy = detector.roberts_edge(image)
prewitt_mag, gx, gy = detector.prewitt_edge(image)
sobel_mag, gx, gy = detector.sobel_edge(image, ksize=3)
canny_edges = detector.canny_edge(image, 50, 150)

# 可视化对比
detector.compare_edge_detectors(image, save_path='comparison.png')
```

## 算子对比

| 算子 | 核大小 | 特点 | 噪声抑制 | 边缘定位 | 速度 |
|------|-------|------|---------|---------|------|
| **Roberts** | 2×2 | 最简单 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Prewitt** | 3×3 | 简单平滑 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Sobel** | 3×3 | 高斯加权 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Canny** | 多步骤 | 最优检测 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |

## API 文档

### 1. 点检测

```python
points = detector.point_detection(image, threshold=0.5)
```

**原理：** 使用拉普拉斯算子检测孤立点
```
[-1 -1 -1]
[-1  8 -1]
[-1 -1 -1]
```

**应用：** 噪声点检测、特征点检测

### 2. 线检测

```python
lines = detector.line_detection(image, direction='horizontal', threshold=0.5)
```

**方向选项：**
- `'horizontal'` - 水平线
- `'vertical'` - 垂直线
- `'diagonal_45'` - 45°对角线
- `'diagonal_135'` - 135°对角线

**应用：** 文档分析、结构检测

### 3. Roberts算子

```python
magnitude, gx, gy = detector.roberts_edge(image)
```

**算子：**
```
Gx = [+1  0]    Gy = [ 0 +1]
     [ 0 -1]         [-1  0]
```

**特点：**
- ✅ 最简单，计算快
- ✅ 边缘定位准确
- ❌ 对噪声敏感
- ❌ 边缘较粗

**适用：** 噪声小的图像，快速检测

### 4. Prewitt算子

```python
magnitude, gx, gy = detector.prewitt_edge(image)
```

**算子：**
```
Gx = [-1  0 +1]    Gy = [-1 -1 -1]
     [-1  0 +1]         [ 0  0  0]
     [-1  0 +1]         [+1 +1 +1]
```

**特点：**
- ✅ 简单均值平滑
- ✅ 对噪声有一定抑制
- ⚠️ 边缘较粗

**适用：** 一般边缘检测

### 5. Sobel算子 ⭐

```python
magnitude, gx, gy = detector.sobel_edge(image, ksize=3)
```

**算子（3×3）：**
```
Gx = [-1  0 +1]    Gy = [-1 -2 -1]
     [-2  0 +2]         [ 0  0  0]
     [-1  0 +1]         [+1 +2 +1]
```

**参数：**
- `ksize`: 核大小（3, 5, 7）

**特点：**
- ✅ 高斯加权，中心权重大
- ✅ 噪声抑制好
- ✅ 边缘定位准确
- ✅ **最常用的边缘检测算子**

**适用：** 标准边缘检测，工业应用

### 6. Canny算子 ⭐⭐

```python
edges = detector.canny_edge(image, low_threshold=50, high_threshold=150)
```

**步骤：**
1. 高斯滤波去噪
2. 计算梯度幅值和方向
3. 非极大值抑制
4. 双阈值检测
5. 边缘跟踪

**参数：**
- `low_threshold`: 低阈值（弱边缘）
- `high_threshold`: 高阈值（强边缘）

**特点：**
- ✅ 最优边缘检测
- ✅ 边缘连续
- ✅ 单像素宽度
- ✅ 噪声抑制最好

**适用：** 高质量边缘检测，物体检测

## 梯度分析

所有算子都返回三个值：
- **magnitude**: 梯度幅值 = √(Gx² + Gy²)
- **gx**: X方向梯度（垂直边缘响应强）
- **gy**: Y方向梯度（水平边缘响应强）

**梯度方向：** θ = arctan(Gy / Gx)

## 实际应用

### 1. 通用边缘检测

```python
# 推荐使用Sobel或Canny
sobel_edges, _, _ = detector.sobel_edge(image, ksize=3)
canny_edges = detector.canny_edge(image, 50, 150)
```

### 2. 噪声图像

```python
# 使用Canny，它有内置的高斯滤波
edges = detector.canny_edge(noisy_image, 50, 150, aperture_size=5)
```

### 3. 快速检测

```python
# 使用Roberts算子
magnitude, _, _ = detector.roberts_edge(image)
```

### 4. 文档线条检测

```python
# 检测水平线和垂直线
h_lines = detector.line_detection(document, 'horizontal', threshold=0.6)
v_lines = detector.line_detection(document, 'vertical', threshold=0.6)

# 合并
grid = cv2.bitwise_or(h_lines, v_lines)
```

### 5. 物体轮廓提取

```python
# 使用Canny获得清晰轮廓
edges = detector.canny_edge(image, 50, 150)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### 6. 方向性边缘

```python
# 只检测垂直边缘
_, gx, _ = detector.sobel_edge(image)
vertical_edges = np.abs(gx).astype(np.uint8)

# 只检测水平边缘
_, _, gy = detector.sobel_edge(image)
horizontal_edges = np.abs(gy).astype(np.uint8)
```

## 算子选择指南

### 按应用场景

| 场景 | 推荐算子 | 原因 |
|------|---------|------|
| **标准边缘检测** | Sobel | 平衡性好 |
| **高质量检测** | Canny | 效果最优 |
| **快速原型** | Roberts | 速度快 |
| **噪声图像** | Canny或Sobel | 噪声抑制好 |
| **实时处理** | Sobel | 速度适中 |

### 按图像特性

| 图像特性 | 推荐算子 |
|---------|---------|
| 噪声小 | Roberts, Prewitt |
| 噪声大 | Sobel, Canny |
| 需要细节 | Canny |
| 计算资源受限 | Roberts |

## 参数调优

### Sobel核大小

```python
# ksize=3: 标准选择
edges_3 = detector.sobel_edge(image, ksize=3)

# ksize=5: 更多平滑，更少噪声
edges_5 = detector.sobel_edge(image, ksize=5)

# ksize=7: 强平滑，可能损失细节
edges_7 = detector.sobel_edge(image, ksize=7)
```

### Canny阈值

```python
# 低阈值：检测更多边缘（包括弱边缘）
edges_sensitive = detector.canny_edge(image, 30, 90)

# 高阈值：只检测强边缘
edges_strong = detector.canny_edge(image, 100, 200)

# 标准设置（high = 2-3 × low）
edges_standard = detector.canny_edge(image, 50, 150)
```

## 技术原理

### 梯度的数学定义

图像梯度：
```
∇f = [∂f/∂x, ∂f/∂y]
```

梯度幅值：
```
|∇f| = √((∂f/∂x)² + (∂f/∂y)²)
```

梯度方向：
```
θ = arctan(∂f/∂y / ∂f/∂x)
```

### Sobel算子的数学推导

Sobel算子 = 高斯平滑 + 差分

X方向：
```
Gx = [1 2 1]ᵀ × [-1 0 1]
```

### Canny算法步骤

1. **高斯滤波**：去除噪声
2. **梯度计算**：使用Sobel算子
3. **非极大值抑制**：细化边缘到单像素
4. **双阈值**：
   - 强边缘：> high_threshold
   - 弱边缘：low_threshold ~ high_threshold
5. **边缘跟踪**：连接弱边缘到强边缘

## 测试结果

```
============================== 25 passed in 0.65s ==============================
```

**测试覆盖：**
- 点检测和线检测
- 4种边缘检测算子
- 梯度方向验证
- 阈值效果测试
- 边界情况

## 常见问题

### Q1: Roberts vs Prewitt vs Sobel 如何选择？

A:
- **Roberts**: 速度最快，边缘定位最准，但对噪声敏感
- **Prewitt**: 中等性能，简单平滑
- **Sobel**: 最常用，噪声抑制好，边缘定位准 ⭐

### Q2: Sobel vs Canny 如何选择？

A:
- **Sobel**: 返回梯度幅值（灰度），计算快，可调性强
- **Canny**: 返回二值边缘，效果最优，边缘连续

**一般选择Sobel**，需要高质量时用Canny

### Q3: Canny的阈值如何设置？

A:
- **经验值**: low=50, high=150
- **自动计算**: 
  ```python
  median = np.median(image)
  low = int(max(0, 0.66 * median))
  high = int(min(255, 1.33 * median))
  ```

### Q4: 为什么我的边缘检测结果很乱？

A: 可能原因：
1. **噪声太多** → 先平滑：`cv2.GaussianBlur(image, (5,5), 1)`
2. **阈值不合适** → 调整阈值
3. **图像对比度低** → 先增强对比度

### Q5: 如何只检测特定方向的边缘？

A: 使用梯度分量：
```python
_, gx, gy = detector.sobel_edge(image)
vertical_edges = np.abs(gx)  # 垂直边缘
horizontal_edges = np.abs(gy)  # 水平边缘
```

## 性能优化

1. **使用OpenCV内置函数**（已实现）
2. **选择合适的算子**：
   - 快速原型 → Roberts
   - 标准应用 → Sobel
   - 高质量 → Canny
3. **降采样**：大图像先缩小
4. **ROI处理**：只处理感兴趣区域

## 许可证

MIT License

## 更新日志

### v1.0.0 (2026-06-04)
- ✅ 实现点检测
- ✅ 实现线检测（4个方向）
- ✅ 实现Roberts边缘检测
- ✅ 实现Prewitt边缘检测
- ✅ 实现Sobel边缘检测
- ✅ 实现Canny边缘检测
- ✅ 25个单元测试全部通过
- ✅ 完整的可视化功能
- ✅ 中文文档
