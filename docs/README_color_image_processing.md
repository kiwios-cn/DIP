# 彩色图像平滑和锐化

## 项目简介

本项目实现了彩色图像的平滑和锐化处理，支持多种颜色空间转换和处理策略。

### 平滑处理
- **高斯平滑**：去除高斯噪声，支持RGB/HSV/LAB颜色空间
- **双边滤波**：保边平滑，美颜效果
- **均值平滑**：简单快速平滑
- **中值平滑**：去除椒盐噪声

### 锐化处理
- **拉普拉斯锐化**：边缘增强，支持多种核类型
- **Unsharp Masking**：专业照片锐化
- **高提升滤波**：细节增强

### 颜色空间
- **RGB**：对三个通道分别处理
- **HSV**：只对V（亮度）通道处理，保持色调和饱和度
- **LAB**：只对L（亮度）通道处理，保持颜色

## 功能特点

1. ✅ **多种平滑算法**：4种平滑方法
2. ✅ **多种锐化算法**：3种锐化方法
3. ✅ **颜色空间支持**：RGB、HSV、LAB
4. ✅ **保边平滑**：双边滤波保持边缘
5. ✅ **专业锐化**：Unsharp Masking照片级效果
6. ✅ **完整测试**：31个单元测试

## 快速开始

```python
from color_image_processing import ColorImageProcessing
import cv2

processor = ColorImageProcessing()
image = cv2.imread('color_image.jpg')

# 平滑处理
smooth_rgb = processor.gaussian_smooth(image, kernel_size=5, color_space='RGB')
smooth_hsv = processor.gaussian_smooth(image, kernel_size=5, color_space='HSV')
bilateral = processor.bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)

# 锐化处理
sharp_lap = processor.laplacian_sharpen(image, alpha=1.0, color_space='RGB')
sharp_unsharp = processor.unsharp_masking(image, amount=1.5, color_space='LAB')
sharp_boost = processor.highboost_filter(image, A=1.5)

# 对比所有方法
processor.compare_smoothing(image, save_path='smoothing.png')
processor.compare_sharpening(image, save_path='sharpening.png')
```

## API 文档

### 平滑处理

#### `gaussian_smooth(image, kernel_size=5, sigma=1.0, color_space='RGB')`
高斯平滑，支持RGB/HSV/LAB颜色空间。

**颜色空间选择：**
- `RGB`：对三通道分别平滑，可能改变颜色
- `HSV`：只平滑V通道，保持色调和饱和度 ⭐
- `LAB`：只平滑L通道，保持颜色信息

#### `bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)`
双边滤波，保边平滑。

**应用：** 美颜、去噪同时保持边缘

#### `mean_smooth(image, kernel_size=5, color_space='RGB')`
均值平滑。

#### `median_smooth(image, kernel_size=5)`
中值平滑，对彩色图像每通道分别处理。

### 锐化处理

#### `laplacian_sharpen(image, kernel_type='4', alpha=1.0, color_space='RGB')`
拉普拉斯锐化。

**参数：**
- `kernel_type`: '4', '8', 'enhanced'
- `alpha`: 锐化强度（0.5-2.0）
- `color_space`: 'RGB', 'HSV', 'LAB'

#### `unsharp_masking(image, kernel_size=5, sigma=1.0, amount=1.5, threshold=0, color_space='RGB')`
Unsharp Masking 专业照片锐化。

**参数：**
- `amount`: 锐化强度（1.0-2.0）
- `threshold`: 只锐化差异大于此值的像素
- `color_space`: 建议用'LAB'保持颜色

#### `highboost_filter(image, kernel_size=5, A=1.5, color_space='RGB')`
高提升滤波。

**参数：**
- `A`: 提升系数（必须>1）

## 颜色空间对比

### RGB vs HSV vs LAB

| 颜色空间 | 处理通道 | 优点 | 缺点 | 推荐场景 |
|---------|---------|------|------|---------|
| **RGB** | B, G, R三通道 | 简单直接 | 可能改变颜色 | 一般处理 |
| **HSV** | 只处理V（亮度） | 保持色调和饱和度 | 亮度变化明显 | 保持颜色 ⭐ |
| **LAB** | 只处理L（亮度） | 颜色保持最好 | 计算稍慢 | 照片处理 ⭐⭐ |

**建议：**
- 平滑：HSV或LAB（保持颜色）
- 锐化：LAB（最佳颜色保持）

## 实际应用

### 1. 照片去噪
```python
# 使用双边滤波保边去噪
result = processor.bilateral_filter(photo, d=9, sigma_color=75, sigma_space=75)
```

### 2. 照片锐化
```python
# 使用Unsharp Masking，LAB空间保持颜色
result = processor.unsharp_masking(photo, amount=1.5, color_space='LAB')
```

### 3. 美颜效果
```python
# 双边滤波 + 保持颜色
result = processor.bilateral_filter(portrait, d=15, sigma_color=80, sigma_space=80)
```

### 4. 扫描文档锐化
```python
# 高提升滤波
result = processor.highboost_filter(document, A=1.8, color_space='LAB')
```

## 技术原理

### 颜色空间转换

**RGB → HSV**
- H (Hue): 色调
- S (Saturation): 饱和度  
- V (Value): 亮度

只处理V通道，H和S保持不变 → 颜色不变

**RGB → LAB**
- L: 亮度 (0-100)
- A: 红绿轴 (-128到127)
- B: 黄蓝轴 (-128到127)

只处理L通道，A和B保持不变 → 颜色最稳定

### Unsharp Masking原理

```
1. 模糊原图: blurred = GaussianBlur(original)
2. 计算掩码: mask = original - blurred
3. 锐化: sharpened = original + amount × mask
```

### 双边滤波

同时考虑空间距离和像素值差异：

```
权重 = exp(-空间距离²/σ_space²) × exp(-像素差²/σ_color²)
```

边缘处像素差大 → 权重小 → 保持边缘

## 测试结果

```
============================== 31 passed in 0.76s ==============================
```

**测试覆盖：**
- 三种颜色空间测试
- 平滑和锐化基本功能
- 参数验证
- 边界情况
- 颜色保持特性

## 常见问题

### Q1: RGB vs HSV vs LAB 如何选择？

A: 
- **需要保持颜色** → HSV或LAB
- **最佳颜色保持** → LAB（推荐）
- **简单快速** → RGB

### Q2: 锐化后颜色失真？

A: 使用LAB颜色空间：
```python
result = processor.unsharp_masking(image, amount=1.5, color_space='LAB')
```

### Q3: 双边滤波参数如何调？

A:
- `d=9`: 标准选择
- `sigma_color`: 50-100（越大平滑越强）
- `sigma_space`: 50-100（越大范围越大）

### Q4: Unsharp Masking vs 拉普拉斯锐化？

A:
- **Unsharp Masking**: 更柔和，适合照片
- **拉普拉斯**: 更锐利，适合文档

## 性能优化

1. **使用OpenCV内置函数**（已实现）
2. **合理选择颜色空间**：LAB转换略慢
3. **批处理**：复用参数处理多张图像

## 许可证

MIT License

## 更新日志

### v1.0.0 (2026-06-03)
- ✅ 实现4种平滑方法
- ✅ 实现3种锐化方法
- ✅ 支持RGB/HSV/LAB颜色空间
- ✅ 31个单元测试全部通过
- ✅ 中文文档
