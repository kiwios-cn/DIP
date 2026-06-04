# 形态学图像处理

## 项目简介

本项目实现了完整的形态学图像处理操作，适用于二值图像和灰度图像。

### 基本操作
- **腐蚀** (Erosion) - 缩小白色区域，去除小白点
- **膨胀** (Dilation) - 扩大白色区域，填充小黑洞

### 复合操作
- **开运算** (Opening) - 先腐蚀后膨胀，去除小白点
- **闭运算** (Closing) - 先膨胀后腐蚀，填充小黑洞

### 高级操作
- **形态学梯度** (Morphological Gradient) - 提取边缘
- **顶帽变换** (Top Hat) - 提取小亮点
- **黑帽变换** (Black Hat) - 提取小暗点

## 功能特点

1. ✅ **7种形态学操作**：基本+复合+高级
2. ✅ **3种结构元素**：矩形、椭圆、十字
3. ✅ **多次迭代支持**：可指定迭代次数
4. ✅ **二值/灰度支持**：同时支持两种图像
5. ✅ **完整测试**：26个单元测试
6. ✅ **可视化对比**：直观展示效果

## 快速开始

```python
from morphological_operations import MorphologicalOperations
import cv2

morph = MorphologicalOperations()
image = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)

# 基本操作
eroded = morph.erosion(image, kernel_size=5)
dilated = morph.dilation(image, kernel_size=5)

# 复合操作
opened = morph.opening(image, kernel_size=5)  # 去除小白点
closed = morph.closing(image, kernel_size=5)  # 填充小黑洞

# 高级操作
gradient = morph.morphological_gradient(image, kernel_size=3)  # 边缘
tophat = morph.top_hat(image, kernel_size=9)  # 小亮点
blackhat = morph.black_hat(image, kernel_size=9)  # 小暗点

# 可视化对比
morph.compare_basic_operations(image, save_path='comparison.png')
```

## API 文档

### 1. 腐蚀 (Erosion)

```python
eroded = morph.erosion(image, kernel_size=3, kernel_shape='rect', iterations=1)
```

**效果：**
- 缩小白色区域（前景）
- 扩大黑色区域（背景）
- 去除小的白色噪声点
- 分离连接的物体

**公式：** `dst = min{src(x+x',y+y') | (x',y')∈kernel}`

### 2. 膨胀 (Dilation)

```python
dilated = morph.dilation(image, kernel_size=3, kernel_shape='rect', iterations=1)
```

**效果：**
- 扩大白色区域
- 缩小黑色区域
- 填充小的黑色孔洞
- 连接邻近的物体

**公式：** `dst = max{src(x+x',y+y') | (x',y')∈kernel}`

### 3. 开运算 (Opening)

```python
opened = morph.opening(image, kernel_size=5, kernel_shape='rect')
```

**效果：**
- 先腐蚀后膨胀
- 去除小于结构元素的白色物体
- 平滑物体轮廓
- 断开细的连接

**公式：** `opening = dilate(erode(src))`

**特点：** 结果 ≤ 原图

### 4. 闭运算 (Closing)

```python
closed = morph.closing(image, kernel_size=5, kernel_shape='rect')
```

**效果：**
- 先膨胀后腐蚀
- 填充小于结构元素的黑色孔洞
- 连接邻近的物体
- 平滑物体轮廓

**公式：** `closing = erode(dilate(src))`

**特点：** 结果 ≥ 原图

### 5. 形态学梯度

```python
gradient = morph.morphological_gradient(image, kernel_size=3)
```

**效果：** 提取物体边缘

**公式：** `gradient = dilate(src) - erode(src)`

### 6. 顶帽变换

```python
tophat = morph.top_hat(image, kernel_size=9)
```

**效果：** 提取比结构元素小的亮区域

**公式：** `tophat = src - opening(src)`

### 7. 黑帽变换

```python
blackhat = morph.black_hat(image, kernel_size=9)
```

**效果：** 提取比结构元素小的暗区域

**公式：** `blackhat = closing(src) - src`

## 结构元素

### 矩形 (rect)
```
1 1 1
1 1 1
1 1 1
```
**特点：** 各向同性，标准选择

### 椭圆 (ellipse)
```
0 1 0
1 1 1
0 1 0
```
**特点：** 圆形，平滑效果好

### 十字 (cross)
```
0 1 0
1 1 1
0 1 0
```
**特点：** 连接性强

## 操作对比

### 腐蚀 vs 膨胀

| 操作 | 白色区域 | 黑色区域 | 应用 |
|------|---------|---------|------|
| 腐蚀 | 缩小 ⬇️ | 扩大 ⬆️ | 去除小白点 |
| 膨胀 | 扩大 ⬆️ | 缩小 ⬇️ | 填充小黑洞 |

### 开运算 vs 闭运算

| 操作 | 公式 | 效果 | 应用 |
|------|------|------|------|
| 开运算 | 腐蚀→膨胀 | 去除小白点 | 噪声去除 |
| 闭运算 | 膨胀→腐蚀 | 填充小黑洞 | 孔洞填充 |

## 实际应用

### 1. 去除椒盐噪声

```python
# 去除白色噪声点（盐噪声）
cleaned = morph.opening(noisy_image, kernel_size=3)

# 填充黑色噪声点（椒噪声）
cleaned = morph.closing(noisy_image, kernel_size=3)

# 去除椒盐混合噪声
temp = morph.opening(noisy_image, 3)
cleaned = morph.closing(temp, 3)
```

### 2. 边缘检测

```python
# 使用形态学梯度
edges = morph.morphological_gradient(image, kernel_size=3)

# 或手动计算
dilated = morph.dilation(image, 3)
eroded = morph.erosion(image, 3)
edges = cv2.subtract(dilated, eroded)
```

### 3. 物体分割

```python
# 分离粘连的物体
separated = morph.erosion(image, kernel_size=5, iterations=2)

# 恢复物体大小
separated = morph.dilation(separated, kernel_size=5, iterations=2)
```

### 4. 特征提取

```python
# 提取亮点
bright_spots = morph.top_hat(image, kernel_size=15)

# 提取暗点
dark_spots = morph.black_hat(image, kernel_size=15)
```

### 5. 骨架提取

```python
# 反复腐蚀直到无法继续
skeleton = image.copy()
element = morph._get_kernel(3, 'cross')

while True:
    eroded = morph.erosion(skeleton, 3, 'cross')
    temp = morph.dilation(eroded, 3, 'cross')
    temp = cv2.subtract(skeleton, temp)
    skeleton = eroded
    if cv2.countNonZero(eroded) == 0:
        break
```

## 技术原理

### 腐蚀的数学定义

设 A 为图像，B 为结构元素：
```
A ⊖ B = {z | B_z ⊆ A}
```
即：B 平移到 z 后完全在 A 内的所有点 z

### 膨胀的数学定义

```
A ⊕ B = {z | (B_z ∩ A) ≠ ∅}
```
即：B 平移到 z 后与 A 有交集的所有点 z

### 对偶性

```
(A ⊖ B)^c = A^c ⊕ B
(A ⊕ B)^c = A^c ⊖ B
```

### 幂等性

```
(A ∘ B) ∘ B = A ∘ B  (开运算)
(A • B) • B = A • B  (闭运算)
```

## 测试结果

```
============================== 26 passed in 0.38s ==============================
```

**测试覆盖：**
- 基本操作（腐蚀、膨胀）
- 复合操作（开、闭）
- 高级操作（梯度、顶帽、黑帽）
- 不同结构元素形状
- 多次迭代
- 幂等性验证
- 边界情况

## 常见问题

### Q1: 腐蚀和膨胀如何选择？

A:
- **去除白色噪声** → 腐蚀或开运算
- **填充黑色孔洞** → 膨胀或闭运算

### Q2: 开运算和闭运算的区别？

A:
- **开运算**：去除小于结构元素的白色物体
- **闭运算**：填充小于结构元素的黑色孔洞

### Q3: 结构元素大小如何选择？

A:
- **3×3**: 轻微效果，保留细节
- **5×5**: 标准选择
- **7×7及以上**: 强效果，可能损失细节

### Q4: 何时使用多次迭代？

A: 需要更强效果时，`iterations=3` 比 `kernel_size=9` 效果更自然

### Q5: 顶帽和黑帽的应用场景？

A:
- **顶帽**：背景不均匀时提取亮点（如文档增强）
- **黑帽**：背景不均匀时提取暗点

## 性能优化

1. **使用OpenCV内置函数**（已实现）
2. **选择合适的结构元素大小**
3. **多次迭代 vs 大结构元素**：
   - `erode(img, 3, iterations=2)` 比 `erode(img, 5)` 更平滑

## 许可证

MIT License

## 更新日志

### v1.0.0 (2026-06-04)
- ✅ 实现腐蚀和膨胀
- ✅ 实现开运算和闭运算
- ✅ 实现形态学梯度
- ✅ 实现顶帽和黑帽变换
- ✅ 支持3种结构元素形状
- ✅ 26个单元测试全部通过
- ✅ 完整的可视化功能
- ✅ 中文文档
