# 高级图像分割

## 项目简介

本项目实现了四种高级图像分割算法：

1. **Canny边缘检测（详细实现）** - 最优边缘检测算法
2. **Hough变换** - 直线检测、圆检测
3. **OTSU最大类间方差法** - 自动阈值分割
4. **分水岭分割算法** - 基于标记的区域分割

## 功能特点

1. ✅ **Canny详细实现**：5个步骤可视化
2. ✅ **Hough变换**：检测直线和圆
3. ✅ **OTSU自动阈值**：无需手动设置阈值
4. ✅ **分水岭分割**：精确的区域分割
5. ✅ **完整测试**：9个单元测试

## 快速开始

```python
from advanced_segmentation import AdvancedSegmentation
import cv2

seg = AdvancedSegmentation()
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Canny边缘检测（详细步骤）
edges, steps = seg.canny_edge_detection(image, 50, 150)

# Hough直线检测
lines = seg.hough_lines(edges, threshold=100)

# Hough圆检测
circles = seg.hough_circles(image, min_radius=10, max_radius=100)

# OTSU自动阈值
threshold, binary = seg.otsu_threshold(image)

# 分水岭分割
color_img = cv2.imread('image.png')
result, markers = seg.watershed_segmentation(color_img, auto_markers=True)

# 可视化Canny步骤
seg.visualize_canny_steps(image, save_path='canny_steps.png')
```

## API 文档

### 1. Canny边缘检测（详细实现）⭐⭐⭐

```python
edges, intermediate = seg.canny_edge_detection(image, low_threshold=50, high_threshold=150, sigma=1.0)
```

**5个步骤：**
1. 高斯滤波去噪
2. 计算梯度幅值和方向（Sobel）
3. 非极大值抑制（细化边缘）
4. 双阈值检测（强/弱边缘）
5. 边缘跟踪（连接弱边缘）

**返回：**
- `edges`: 最终边缘图像
- `intermediate`: 包含所有中间步骤的字典

**特点：**
- 最优边缘检测算法
- 边缘连续且单像素宽
- 噪声抑制最好

### 2. Hough变换

#### 直线检测

```python
lines = seg.hough_lines(edges, rho=1, theta=np.pi/180, threshold=100, 
                        min_line_length=50, max_line_gap=10)
```

**原理：** 直线的极坐标表示 ρ = x·cos(θ) + y·sin(θ)

**返回：** `[[x1,y1,x2,y2], ...]` 线段列表

#### 圆检测

```python
circles = seg.hough_circles(image, min_radius=10, max_radius=100, 
                           param1=100, param2=30, min_dist=50)
```

**原理：** 圆的方程 (x-a)² + (y-b)² = r²

**返回：** `[[x,y,r], ...]` 圆列表

### 3. OTSU最大类间方差法

```python
threshold, binary = seg.otsu_threshold(image)
```

**原理：** 自动计算最优阈值，使类间方差最大

**公式：**
```
σ²_between = w0 × w1 × (μ0 - μ1)²
```

**优点：**
- 自动确定阈值
- 适合双峰直方图
- 无需人工调参

### 4. 分水岭分割

```python
result, markers = seg.watershed_segmentation(image, auto_markers=True)
```

**原理：** 将图像视为地形图，从标记点"注水"

**步骤：**
1. OTSU二值化
2. 形态学操作去噪
3. 距离变换找前景
4. 标记连通区域
5. 应用分水岭算法

**应用：** 分离粘连物体、区域分割

## 算法对比

| 算法 | 类型 | 输入 | 输出 | 适用场景 |
|------|------|------|------|---------|
| **Canny** | 边缘检测 | 灰度图 | 二值边缘 | 通用边缘检测 |
| **Hough** | 形状检测 | 边缘图 | 直线/圆参数 | 结构化物体检测 |
| **OTSU** | 阈值分割 | 灰度图 | 二值图像 | 前景背景分离 |
| **分水岭** | 区域分割 | 彩色图 | 分割区域 | 粘连物体分离 |

## 实际应用

### 1. 边缘检测管道

```python
# 完整的边缘检测流程
edges, steps = seg.canny_edge_detection(image, 50, 150)

# 查看中间步骤
blurred = steps['blurred']
magnitude = steps['magnitude']
nms = steps['nms']
```

### 2. 直线检测（文档分析）

```python
# 检测文档中的表格线
edges, _ = seg.canny_edge_detection(document, 50, 150)
lines = seg.hough_lines(edges, threshold=100, min_line_length=100)

# 绘制检测到的直线
for x1, y1, x2, y2 in lines:
    cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

### 3. 圆检测（硬币计数）

```python
# 检测硬币
circles = seg.hough_circles(coins_image, min_radius=20, max_radius=50)

print(f"检测到 {len(circles)} 个硬币")
for x, y, r in circles:
    cv2.circle(result, (x, y), r, (0, 255, 0), 2)
```

### 4. 自动阈值分割

```python
# 无需手动设置阈值
threshold, binary = seg.otsu_threshold(image)
print(f"最优阈值: {threshold}")
```

### 5. 分离粘连物体

```python
# 使用分水岭分离粘连的细胞
result, markers = seg.watershed_segmentation(cell_image, auto_markers=True)
```

## Canny算法详解

### 步骤1：高斯滤波
去除噪声，平滑图像

### 步骤2：梯度计算
使用Sobel算子计算梯度幅值和方向

### 步骤3：非极大值抑制 ⭐
沿梯度方向，保留局部最大值，抑制非最大值
- 将边缘细化到单像素宽度

### 步骤4：双阈值
- 强边缘：magnitude > high_threshold
- 弱边缘：low_threshold < magnitude < high_threshold

### 步骤5：边缘跟踪
连接弱边缘到强边缘，形成连续边缘

## OTSU算法原理

### 类间方差最大化

遍历所有可能的阈值t，计算：

```
w0 = 前景像素比例
w1 = 背景像素比例
μ0 = 前景平均灰度
μ1 = 背景平均灰度

σ²_between(t) = w0 × w1 × (μ0 - μ1)²
```

选择使 σ²_between 最大的t作为最优阈值

## 分水岭算法原理

### 地形图类比

1. 将梯度图像视为地形图（梯度值=高度）
2. 从标记点开始向上"注水"
3. 不同区域的水汇合处=分水岭（边界）

### 标记生成

- 前景标记：距离变换的局部最大值
- 背景标记：确定的背景区域
- 不确定区域：前景和背景之间

## 参数调优

### Canny阈值

```python
# 低阈值：检测更多边缘
edges = seg.canny_edge_detection(img, 30, 90)

# 高阈值：只保留强边缘
edges = seg.canny_edge_detection(img, 100, 200)

# 经验值：high = 2-3 × low
edges = seg.canny_edge_detection(img, 50, 150)
```

### Hough阈值

```python
# 低阈值：检测更多直线（包括短线）
lines = seg.hough_lines(edges, threshold=50)

# 高阈值：只检测显著直线
lines = seg.hough_lines(edges, threshold=200)
```

## 测试结果

```
============================== 9 passed in 0.76s ==============================
```

## 许可证

MIT License

## 更新日志

### v1.0.0 (2026-06-04)
- ✅ 实现Canny边缘检测（详细5步）
- ✅ 实现Hough直线检测
- ✅ 实现Hough圆检测
- ✅ 实现OTSU最大类间方差法
- ✅ 实现分水岭分割算法
- ✅ 9个单元测试全部通过
- ✅ Canny步骤可视化
- ✅ 中文文档
