# 均值滤波器

## 项目简介

本项目实现了四种经典的均值滤波器，用于图像去噪和平滑处理：

1. **算术均值滤波器** (Arithmetic Mean Filter)
2. **几何均值滤波器** (Geometric Mean Filter)  
3. **谐波均值滤波器** (Harmonic Mean Filter)
4. **逆谐波均值滤波器** (Contraharmonic Mean Filter)

每种滤波器针对不同类型的噪声有不同的效果，是数字图像处理中的基础工具。

## 功能特点

1. ✅ **4种均值滤波器**：覆盖不同应用场景
2. ✅ **灵活参数控制**：卷积核大小、Q值可调
3. ✅ **边界处理**：自动处理图像边界
4. ✅ **零值保护**：几何均值和谐波均值自动处理0值
5. ✅ **完整测试**：28个单元测试，覆盖所有功能
6. ✅ **可视化对比**：直观展示不同滤波器效果

## 文件结构

```
.
├── mean_filters.py                    # 主程序文件
├── test_mean_filters.py               # 测试文件
├── README_mean_filters.md             # 本文档
├── mean_filters_gaussian_comparison.png   # 高斯噪声滤波对比
├── mean_filters_sp_comparison.png         # 椒盐噪声滤波对比
├── mean_filters_salt_comparison.png       # 盐噪声滤波对比
├── mean_filters_pepper_comparison.png     # 椒噪声滤波对比
└── mean_filters_Q_effect.png              # Q值效果对比
```

## 依赖项

```bash
pip install opencv-python numpy matplotlib pytest
```

## 快速开始

### 1. 基本使用

```python
from mean_filters import MeanFilters
import cv2

# 创建滤波器对象
filters = MeanFilters()

# 加载图像
image = cv2.imread('noisy_image.png', cv2.IMREAD_GRAYSCALE)

# 算术均值滤波
result = filters.arithmetic_mean_filter(image, kernel_size=5)

# 几何均值滤波
result = filters.geometric_mean_filter(image, kernel_size=5)

# 谐波均值滤波
result = filters.harmonic_mean_filter(image, kernel_size=5)

# 逆谐波均值滤波
result = filters.contraharmonic_mean_filter(image, kernel_size=5, Q=1.5)
```

### 2. 对比所有滤波器

```python
# 可视化所有滤波器效果
filters.compare_filters(
    noisy_image,
    kernel_size=5,
    Q_positive=1.5,   # 去除椒噪声
    Q_negative=-1.5,  # 去除盐噪声
    save_path='comparison.png'
)
```

### 3. 演示Q值效果

```python
# 展示不同Q值的效果
filters.demonstrate_Q_effect(
    noisy_image,
    kernel_size=5,
    save_path='Q_effect.png'
)
```

### 4. 运行演示程序

```bash
python mean_filters.py
```

这将：
1. 测试高斯噪声去除
2. 测试椒盐噪声去除
3. 测试盐噪声去除
4. 测试椒噪声去除
5. 演示Q值效果

## 运行测试

```bash
# 运行所有测试
python -m pytest test_mean_filters.py -v

# 查看测试覆盖率
python -m pytest test_mean_filters.py --cov=mean_filters --cov-report=html
```

**测试结果：** 28/28 通过 ✅

## API 文档

### MeanFilters 类

#### 1. 算术均值滤波器

##### `arithmetic_mean_filter(image, kernel_size=3)`

对邻域内所有像素求算术平均值。

**公式：**
```
f_hat(x,y) = (1/mn) × Σ g(s,t)
```

**参数：**
- `image` (np.ndarray): 输入图像
- `kernel_size` (int): 卷积核大小（奇数），默认3

**返回：** `np.ndarray` - 滤波后的图像

**应用场景：**
- ✅ 去除高斯噪声
- ✅ 图像平滑
- ✅ 预处理

**优点：**
- 简单高效
- 对高斯噪声效果好
- 计算速度快

**缺点：**
- 会模糊图像边缘
- 对椒盐噪声效果差

**示例：**
```python
# 3×3算术均值滤波
result = filters.arithmetic_mean_filter(noisy_image, kernel_size=3)

# 5×5算术均值滤波（更强的平滑）
result = filters.arithmetic_mean_filter(noisy_image, kernel_size=5)
```

---

#### 2. 几何均值滤波器

##### `geometric_mean_filter(image, kernel_size=3)`

对邻域内所有像素求几何平均值。平滑效果介于算术均值和中值之间。

**公式：**
```
f_hat(x,y) = [Π g(s,t)]^(1/mn)
```

**参数：**
- `image` (np.ndarray): 输入图像
- `kernel_size` (int): 卷积核大小（奇数），默认3

**返回：** `np.ndarray` - 滤波后的图像

**应用场景：**
- ✅ 去除高斯噪声
- ✅ 保持边缘细节
- ✅ 需要较好边缘保持的平滑

**优点：**
- 比算术均值更好地保持边缘
- 对高斯噪声效果好
- 平滑效果适中

**缺点：**
- 计算复杂度高于算术均值
- 需要特殊处理0值

**实现细节：**
- 使用对数运算避免数值溢出
- `geometric_mean = exp(mean(log(x)))`
- 添加小量避免log(0)

**示例：**
```python
# 几何均值滤波（保持边缘）
result = filters.geometric_mean_filter(noisy_image, kernel_size=5)
```

---

#### 3. 谐波均值滤波器

##### `harmonic_mean_filter(image, kernel_size=3)`

对邻域内所有像素求谐波平均值。特别适合处理盐噪声（亮点）。

**公式：**
```
f_hat(x,y) = mn / Σ(1/g(s,t))
```

**参数：**
- `image` (np.ndarray): 输入图像
- `kernel_size` (int): 卷积核大小（奇数），默认3

**返回：** `np.ndarray` - 滤波后的图像

**应用场景：**
- ✅✅ 去除盐噪声（白点）- **最佳选择**
- ✅ 处理高亮异常点
- ✅ 保持暗区域细节

**优点：**
- 对盐噪声（亮点）效果好
- 能保持暗区域细节
- 对高亮异常值有抑制作用

**缺点：**
- 对椒噪声（黑点）效果差甚至恶化
- 会使图像整体变暗
- 需要特殊处理0值

**注意事项：**
- 不适合处理椒噪声（黑点）
- 对0值会产生问题，需要避免除零

**示例：**
```python
# 去除盐噪声（白点）
result = filters.harmonic_mean_filter(salt_noisy_image, kernel_size=5)
```

---

#### 4. 逆谐波均值滤波器 ⭐

##### `contraharmonic_mean_filter(image, kernel_size=3, Q=1.0)`

对邻域内像素进行加权平均，权重由阶数Q控制。最灵活的均值滤波器。

**公式：**
```
f_hat(x,y) = Σ g(s,t)^(Q+1) / Σ g(s,t)^Q
```

**参数：**
- `image` (np.ndarray): 输入图像
- `kernel_size` (int): 卷积核大小（奇数），默认3
- `Q` (float): 阶数（order）
  - `Q > 0`: 去除椒噪声（黑点）
  - `Q < 0`: 去除盐噪声（白点）
  - `Q = 0`: 等同于算术均值
  - `Q = -1`: 等同于谐波均值

**返回：** `np.ndarray` - 滤波后的图像

**应用场景：**

| Q值范围 | 效果 | 应用 |
|---------|------|------|
| Q = 0 | 算术均值 | 高斯噪声 |
| Q = -1 | 谐波均值 | 盐噪声 |
| **Q = 1.5** | **强力去除椒噪声** | **黑点噪声** ✅✅ |
| **Q = -1.5** | **强力去除盐噪声** | **白点噪声** ✅✅ |
| Q > 2 | 过度校正 | 不推荐 |
| Q < -2 | 过度校正 | 不推荐 |

**优点：**
- 灵活性高，可针对性处理不同噪声
- Q > 0 时对椒噪声效果好
- Q < 0 时对盐噪声效果好
- 统一了算术均值和谐波均值

**缺点：**
- Q选择不当会产生负面效果
- 计算复杂度较高
- 需要根据噪声类型调整Q值

**Q值选择指南：**

```python
# 去除椒噪声（黑点）
result = filters.contraharmonic_mean_filter(pepper_noisy, kernel_size=5, Q=1.5)

# 去除盐噪声（白点）
result = filters.contraharmonic_mean_filter(salt_noisy, kernel_size=5, Q=-1.5)

# 去除高斯噪声（等同于算术均值）
result = filters.contraharmonic_mean_filter(gaussian_noisy, kernel_size=5, Q=0)

# 轻微去除椒噪声
result = filters.contraharmonic_mean_filter(image, kernel_size=3, Q=0.5)

# 轻微去除盐噪声  
result = filters.contraharmonic_mean_filter(image, kernel_size=3, Q=-0.5)
```

---

#### 辅助方法

##### `compare_filters(image, kernel_size=3, Q_positive=1.5, Q_negative=-1.5, save_path=None)`

比较所有滤波器效果。

**生成对比图包含：**
- 原始图像
- 算术均值滤波
- 几何均值滤波
- 谐波均值滤波
- 逆谐波均值滤波 (Q > 0)
- 逆谐波均值滤波 (Q < 0)

##### `demonstrate_Q_effect(image, kernel_size=3, save_path=None)`

演示逆谐波滤波器不同Q值的效果。

**展示Q值：** -2, -1, 0, 1, 2

## 滤波器特性对比

### 数学关系

在均匀图像上，四种滤波器有以下关系：

```
谐波均值 ≤ 几何均值 ≤ 算术均值
```

### 适用噪声类型

| 滤波器 | 高斯噪声 | 盐噪声（白点） | 椒噪声（黑点） | 椒盐混合 |
|--------|---------|--------------|--------------|---------|
| 算术均值 | ✅✅ | ❌ | ❌ | ❌ |
| 几何均值 | ✅✅ | ⚠️ | ⚠️ | ⚠️ |
| 谐波均值 | ✅ | ✅✅ | ❌❌ | ❌ |
| 逆谐波 (Q>0) | ✅ | ❌ | ✅✅ | ⚠️ |
| 逆谐波 (Q<0) | ✅ | ✅✅ | ❌ | ⚠️ |

**图例：** ✅✅ 最佳 | ✅ 有效 | ⚠️ 一般 | ❌ 不适用 | ❌❌ 会恶化

### 边缘保持能力

| 滤波器 | 边缘保持 | 平滑程度 | 计算复杂度 |
|--------|---------|---------|----------|
| 算术均值 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| 几何均值 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 谐波均值 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 逆谐波均值 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

## 实际应用示例

### 1. 去除高斯噪声

```python
from mean_filters import MeanFilters
import cv2

filters = MeanFilters()
noisy_image = cv2.imread('gaussian_noisy.png', cv2.IMREAD_GRAYSCALE)

# 方法1：算术均值（最常用）
result1 = filters.arithmetic_mean_filter(noisy_image, kernel_size=5)

# 方法2：几何均值（更好的边缘保持）
result2 = filters.geometric_mean_filter(noisy_image, kernel_size=5)

# 方法3：逆谐波均值 (Q=0)
result3 = filters.contraharmonic_mean_filter(noisy_image, kernel_size=5, Q=0)

cv2.imwrite('denoised.png', result1)
```

### 2. 去除盐噪声（只有白点）

```python
# 盐噪声：图像中有很多白色亮点

# 方法1：谐波均值（推荐）
result1 = filters.harmonic_mean_filter(salt_noisy, kernel_size=5)

# 方法2：逆谐波均值 Q<0
result2 = filters.contraharmonic_mean_filter(salt_noisy, kernel_size=5, Q=-1.5)

cv2.imwrite('salt_removed.png', result1)
```

### 3. 去除椒噪声（只有黑点）

```python
# 椒噪声：图像中有很多黑色暗点

# 使用逆谐波均值 Q>0（唯一有效方法）
result = filters.contraharmonic_mean_filter(pepper_noisy, kernel_size=5, Q=1.5)

cv2.imwrite('pepper_removed.png', result)
```

### 4. 去除椒盐混合噪声

```python
# 椒盐混合噪声：既有黑点又有白点

# 方法1：先去椒，再去盐（推荐）
temp = filters.contraharmonic_mean_filter(sp_noisy, kernel_size=3, Q=1.5)
result = filters.contraharmonic_mean_filter(temp, kernel_size=3, Q=-1.5)

# 方法2：中值滤波（参考 spatial_filtering.py）
# result = filters.median_filter(sp_noisy, kernel_size=5)

cv2.imwrite('sp_removed.png', result)
```

### 5. 医学图像去噪

```python
# 医学图像（X光、CT等）通常有高斯噪声

medical_image = cv2.imread('xray.png', cv2.IMREAD_GRAYSCALE)

# 使用几何均值（保持边缘）
denoised = filters.geometric_mean_filter(medical_image, kernel_size=3)

cv2.imwrite('xray_denoised.png', denoised)
```

### 6. 扫描文档去噪

```python
# 扫描文档可能有盐噪声（白点）

document = cv2.imread('scanned_doc.png', cv2.IMREAD_GRAYSCALE)

# 使用谐波均值去除白点
clean_doc = filters.harmonic_mean_filter(document, kernel_size=3)

cv2.imwrite('doc_clean.png', clean_doc)
```

### 7. 批量处理

```python
import glob

# 批量去噪
image_files = glob.glob('noisy_images/*.png')

for img_path in image_files:
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 自动选择滤波器
    # 这里假设都是高斯噪声
    denoised = filters.arithmetic_mean_filter(image, kernel_size=5)
    
    output_path = img_path.replace('noisy_images', 'clean_images')
    cv2.imwrite(output_path, denoised)

print(f"处理完成 {len(image_files)} 张图像")
```

## 技术原理

### 均值的数学定义

对于邻域 S_{xy} 内的像素 g(s,t)：

#### 算术均值
```
AM = (1/n) × Σ x_i
```

#### 几何均值
```
GM = (Π x_i)^(1/n)
```

#### 谐波均值
```
HM = n / Σ(1/x_i)
```

#### 逆谐波均值
```
CHM = Σ x_i^(Q+1) / Σ x_i^Q
```

### 数学不等式

对于正数序列，有以下关系：

```
谐波均值 ≤ 几何均值 ≤ 算术均值
HM ≤ GM ≤ AM
```

**证明：**
- HM ≤ GM: 柯西不等式
- GM ≤ AM: 算术-几何平均不等式

### 逆谐波均值的特殊性质

- `Q = 0`: `CHM = AM` (算术均值)
- `Q = -1`: `CHM = HM` (谐波均值)
- `Q → ∞`: 趋向于最大值
- `Q → -∞`: 趋向于最小值

**对噪声的影响：**
- `Q > 0`: 增大高值，抑制低值 → 去除黑点
- `Q < 0`: 增大低值，抑制高值 → 去除白点

## 性能优化

1. **使用OpenCV内置函数**
   - 算术均值使用 `cv2.blur()`
   - 比纯Python实现快50-100倍

2. **几何均值优化**
   - 使用对数运算避免溢出
   - `exp(mean(log(x)))` 代替 `(Π x)^(1/n)`

3. **合理选择卷积核**
   - 3×3: 轻度平滑，速度最快
   - 5×5: 标准选择
   - 7×7及以上: 强平滑，计算量大

4. **批处理优化**
   - 复用滤波器对象
   - 使用相同参数时可以缓存卷积核

## 常见问题

### Q1: 如何选择合适的滤波器？

A: 根据噪声类型选择：

| 噪声类型 | 推荐滤波器 | 参数 |
|---------|-----------|------|
| 高斯噪声 | 算术均值 | kernel_size=5 |
| 高斯噪声+保边 | 几何均值 | kernel_size=5 |
| 盐噪声（白点） | 谐波均值 或 逆谐波(Q=-1.5) | kernel_size=5 |
| 椒噪声（黑点） | 逆谐波(Q=1.5) | kernel_size=5 |
| 椒盐混合 | 中值滤波（见spatial_filtering） | kernel_size=5 |

### Q2: 逆谐波滤波器的Q值如何选择？

A: 
- **轻微噪声**: `|Q| = 0.5-1.0`
- **中等噪声**: `|Q| = 1.0-1.5` (推荐)
- **严重噪声**: `|Q| = 1.5-2.0`
- **不推荐**: `|Q| > 2.0` (可能过度校正)

### Q3: 为什么谐波均值不能去除椒噪声？

A: 谐波均值公式：`HM = n / Σ(1/x_i)`

- 盐噪声(255): `1/255` 很小，对分母贡献小 → 抑制白点 ✅
- 椒噪声(0): `1/0` 趋向无穷，分母很大 → HM趋向0 → **使黑点扩散** ❌

### Q4: 几何均值为什么比算术均值保边效果好？

A: 
- **算术均值**: 对所有值一视同仁
- **几何均值**: 对极值更敏感，受低值影响大

在边缘处（一侧亮，一侧暗），几何均值更接近暗侧，因此边缘模糊较少。

### Q5: 滤波核大小如何选择？

A:
- **3×3**: 轻度平滑，保持细节，速度快
- **5×5**: 标准选择，平衡效果和速度
- **7×7**: 强平滑，模糊较明显
- **9×9及以上**: 过度平滑，不推荐用于去噪

### Q6: 能处理彩色图像吗？

A: 可以，但需要对每个通道分别处理：

```python
# 分离通道
b, g, r = cv2.split(color_image)

# 分别滤波
b_filtered = filters.arithmetic_mean_filter(b, kernel_size=5)
g_filtered = filters.arithmetic_mean_filter(g, kernel_size=5)
r_filtered = filters.arithmetic_mean_filter(r, kernel_size=5)

# 合并通道
result = cv2.merge([b_filtered, g_filtered, r_filtered])
```

### Q7: 滤波后图像整体变暗/变亮？

A: 这是正常现象：
- **谐波均值**: 会使图像变暗（抑制高值）
- **逆谐波 Q>0**: 会使图像变亮（增强高值）
- **逆谐波 Q<0**: 会使图像变暗（增强低值）

如果需要保持亮度，可以在滤波后进行亮度归一化。

## 与其他滤波器的比较

### vs 中值滤波

| 特性 | 均值滤波器 | 中值滤波器 |
|-----|----------|-----------|
| 类型 | 线性（除几何） | 非线性 |
| 高斯噪声 | ✅✅ | ⚠️ |
| 椒盐噪声 | ⚠️ (需选择合适类型) | ✅✅ |
| 边缘保持 | ⚠️ | ✅ |
| 计算速度 | ✅ | ⚠️ |

**建议：**
- 高斯噪声 → 均值滤波
- 椒盐噪声 → 中值滤波
- 混合噪声 → 先中值，再均值

### vs 高斯滤波

| 特性 | 算术均值滤波 | 高斯滤波 |
|-----|------------|---------|
| 权重 | 均匀 | 高斯分布 |
| 边缘保持 | ⚠️ | ✅ |
| 计算速度 | ✅✅ | ✅ |
| 效果 | 一般平滑 | 更自然的平滑 |

**建议：**
- 简单快速去噪 → 算术均值
- 高质量平滑 → 高斯滤波

## 扩展功能建议

1. **自适应滤波**
   - 根据局部统计特性自动选择滤波器
   - 根据局部方差调整卷积核大小

2. **加权滤波**
   - 引入空间权重
   - 结合高斯权重

3. **多尺度滤波**
   - 金字塔多尺度处理
   - 多尺度融合

4. **彩色图像优化**
   - YCbCr空间处理
   - 只对亮度通道滤波

## 许可证

MIT License

## 作者

数字图像处理项目

## 更新日志

### v1.0.0 (2026-06-03)
- ✅ 实现算术均值滤波器
- ✅ 实现几何均值滤波器
- ✅ 实现谐波均值滤波器
- ✅ 实现逆谐波均值滤波器
- ✅ 28个单元测试，全部通过
- ✅ 完整的可视化功能
- ✅ 中文文档
