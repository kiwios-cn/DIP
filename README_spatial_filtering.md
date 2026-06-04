# 空间域滤波实现

## 项目简介

本项目实现了完整的空间域滤波功能，包括：

### 均值滤波
- **标准均值滤波**：算术平均，适合去除高斯噪声
- **加权均值滤波**：高斯加权，保持边缘的同时平滑
- **自定义卷积核**：支持任意卷积核的均值滤波

### 中值滤波
- **标准中值滤波**：非线性滤波，有效去除椒盐噪声
- **自适应中值滤波**：根据局部统计特性自适应调整窗口大小

### 拉普拉斯滤波
- **4邻域拉普拉斯**：基本边缘检测
- **8邻域拉普拉斯**：增强边缘检测
- **增强型拉普拉斯**：对角线加权的拉普拉斯
- **拉普拉斯锐化**：图像锐化增强
- **LoG滤波器**：高斯拉普拉斯，先平滑再检测边缘

## 功能特点

1. **完整的滤波器实现**：涵盖平滑、去噪、边缘检测、锐化
2. **多种噪声模拟**：高斯噪声、椒盐噪声
3. **灵活的参数控制**：卷积核大小、滤波强度可调
4. **边界处理**：自动处理图像边界
5. **完整测试**：26个单元测试，覆盖所有功能
6. **可视化展示**：对比展示所有滤波效果

## 文件结构

```
.
├── spatial_filtering.py              # 主程序文件
├── test_spatial_filtering.py         # 测试文件
├── README_spatial_filtering.md       # 本文档
├── spatial_filtering_results.png     # 滤波效果演示
└── test_spatial_filtering.png        # 测试图像
```

## 依赖项

```bash
pip install opencv-python numpy matplotlib scipy pytest
```

## 使用方法

### 1. 基本使用

```python
from spatial_filtering import SpatialFiltering
import cv2

# 创建滤波器对象
filter_obj = SpatialFiltering()

# 加载图像
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# 均值滤波
mean_result = filter_obj.mean_filter(image, kernel_size=3)

# 中值滤波
median_result = filter_obj.median_filter(image, kernel_size=3)

# 拉普拉斯滤波
laplacian_result = filter_obj.laplacian_filter(image, kernel_type='4')

# 拉普拉斯锐化
sharpened = filter_obj.laplacian_sharpen(image, kernel_type='4', alpha=1.0)

# 保存结果
cv2.imwrite('result.png', mean_result)
```

### 2. 添加噪声并去噪

```python
# 添加高斯噪声
noisy = filter_obj.add_noise(image, 'gaussian', sigma=25)

# 使用均值滤波去噪
denoised = filter_obj.mean_filter(noisy, kernel_size=5)

# 添加椒盐噪声
sp_noisy = filter_obj.add_noise(image, 'salt_pepper', prob=0.05)

# 使用中值滤波去噪
denoised_sp = filter_obj.median_filter(sp_noisy, kernel_size=3)
```

### 3. 自适应中值滤波

```python
# 对于密集的椒盐噪声，使用自适应中值滤波
adaptive_result = filter_obj.adaptive_median_filter(
    sp_noisy, 
    max_kernel_size=7
)
```

### 4. LoG边缘检测

```python
# LoG滤波器（先高斯平滑，再拉普拉斯）
log_result = filter_obj.log_filter(image, sigma=1.5)

# 取绝对值用于边缘显示
edges = np.abs(log_result).astype(np.uint8)
```

### 5. 可视化所有滤波器

```python
# 展示所有滤波效果
filter_obj.visualize_all_filters(
    image,
    save_path='all_filters.png'
)
```

### 6. 运行演示程序

```bash
python spatial_filtering.py
```

这将：
1. 创建测试图像
2. 应用所有滤波器
3. 生成对比图 `spatial_filtering_results.png`

## 运行测试

```bash
# 运行所有测试
python -m pytest test_spatial_filtering.py -v

# 运行特定测试
python -m pytest test_spatial_filtering.py::TestSpatialFiltering::test_mean_filter_basic -v

# 查看测试覆盖率
python -m pytest test_spatial_filtering.py --cov=spatial_filtering --cov-report=html
```

**测试结果：** 26/26 通过 ✅

## API 文档

### SpatialFiltering 类

#### 均值滤波方法

##### `mean_filter(image, kernel_size=3)`
标准均值滤波（算术平均）。

**参数：**
- `image` (np.ndarray): 输入图像
- `kernel_size` (int): 卷积核大小（奇数），默认3

**返回：**
- `np.ndarray`: 滤波后的图像

**应用场景：**
- 去除高斯噪声
- 图像平滑
- 预处理

**示例：**
```python
# 3×3均值滤波
result = filter_obj.mean_filter(image, kernel_size=3)

# 5×5均值滤波（更强的平滑）
result = filter_obj.mean_filter(image, kernel_size=5)
```

##### `weighted_mean_filter(image, kernel_size=3)`
加权均值滤波（高斯加权）。

**参数：**
- `image` (np.ndarray): 输入图像
- `kernel_size` (int): 卷积核大小（奇数），默认3

**返回：**
- `np.ndarray`: 滤波后的图像

**应用场景：**
- 保持边缘的平滑
- 去除高斯噪声
- 比标准均值滤波效果更好

##### `custom_mean_filter(image, kernel)`
自定义卷积核的均值滤波。

**参数：**
- `image` (np.ndarray): 输入图像
- `kernel` (np.ndarray): 自定义卷积核

**返回：**
- `np.ndarray`: 滤波后的图像

**示例：**
```python
# 创建自定义卷积核
kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]], dtype=np.float32)

result = filter_obj.custom_mean_filter(image, kernel)
```

#### 中值滤波方法

##### `median_filter(image, kernel_size=3)`
标准中值滤波。

**参数：**
- `image` (np.ndarray): 输入图像
- `kernel_size` (int): 卷积核大小（奇数），默认3

**返回：**
- `np.ndarray`: 滤波后的图像

**应用场景：**
- 去除椒盐噪声（最有效）
- 保持边缘
- 非线性滤波

**示例：**
```python
# 去除椒盐噪声
result = filter_obj.median_filter(noisy_image, kernel_size=3)
```

##### `adaptive_median_filter(image, max_kernel_size=7)`
自适应中值滤波。

**参数：**
- `image` (np.ndarray): 输入图像
- `max_kernel_size` (int): 最大窗口大小，默认7

**返回：**
- `np.ndarray`: 滤波后的图像

**应用场景：**
- 密集椒盐噪声
- 保持更多细节
- 自适应调整窗口

**工作原理：**
1. 从小窗口开始
2. 判断中值是否为噪声
3. 如果是噪声，增大窗口
4. 直到找到有效中值或达到最大窗口

#### 拉普拉斯滤波方法

##### `laplacian_filter(image, kernel_type='4')`
拉普拉斯滤波。

**参数：**
- `image` (np.ndarray): 输入图像
- `kernel_type` (str): 卷积核类型
  - `'4'` - 4邻域拉普拉斯
  - `'8'` - 8邻域拉普拉斯
  - `'enhanced'` - 增强型拉普拉斯

**返回：**
- `np.ndarray`: 拉普拉斯响应（float32，可能有负值）

**应用场景：**
- 边缘检测
- 图像锐化
- 特征提取

**卷积核：**
```python
# 4邻域
[[0,  1,  0],
 [1, -4,  1],
 [0,  1,  0]]

# 8邻域
[[1,  1,  1],
 [1, -8,  1],
 [1,  1,  1]]

# 增强型
[[1,  4,  1],
 [4, -20, 4],
 [1,  4,  1]]
```

##### `laplacian_sharpen(image, kernel_type='4', alpha=1.0)`
拉普拉斯锐化。

**参数：**
- `image` (np.ndarray): 输入图像
- `kernel_type` (str): 拉普拉斯核类型
- `alpha` (float): 锐化强度，默认1.0

**返回：**
- `np.ndarray`: 锐化后的图像

**公式：**
```
锐化图像 = 原图像 - alpha × 拉普拉斯响应
```

**应用场景：**
- 图像锐化
- 增强细节
- 提高清晰度

**示例：**
```python
# 弱锐化
result = filter_obj.laplacian_sharpen(image, alpha=0.5)

# 标准锐化
result = filter_obj.laplacian_sharpen(image, alpha=1.0)

# 强锐化
result = filter_obj.laplacian_sharpen(image, alpha=2.0)
```

##### `log_filter(image, sigma=1.0)`
LoG滤波器（Laplacian of Gaussian）。

**参数：**
- `image` (np.ndarray): 输入图像
- `sigma` (float): 高斯标准差，默认1.0

**返回：**
- `np.ndarray`: LoG响应

**应用场景：**
- 边缘检测
- 斑点检测
- 特征提取

**工作原理：**
1. 先高斯平滑（去噪）
2. 再拉普拉斯（检测边缘）

#### 辅助方法

##### `add_noise(image, noise_type='gaussian', **kwargs)`
添加噪声。

**参数：**
- `image` (np.ndarray): 输入图像
- `noise_type` (str): 噪声类型
  - `'gaussian'` - 高斯噪声
  - `'salt_pepper'` - 椒盐噪声
- `**kwargs`: 噪声参数
  - 高斯噪声：`mean=0, sigma=25`
  - 椒盐噪声：`prob=0.05`

**返回：**
- `np.ndarray`: 添加噪声后的图像

**示例：**
```python
# 添加高斯噪声
noisy = filter_obj.add_noise(image, 'gaussian', sigma=25)

# 添加椒盐噪声
noisy = filter_obj.add_noise(image, 'salt_pepper', prob=0.05)
```

##### `load_image(path)`
加载图像。

**参数：**
- `path` (str): 图像路径

**返回：**
- `bool`: 是否成功加载

##### `visualize_all_filters(image, save_path=None)`
可视化所有滤波器效果。

**参数：**
- `image` (np.ndarray): 输入图像
- `save_path` (str, optional): 保存路径

## 实际应用示例

### 1. 医学图像去噪

```python
# 加载医学图像
medical_image = cv2.imread('xray.png', cv2.IMREAD_GRAYSCALE)

# 使用加权均值滤波去除高斯噪声
denoised = filter_obj.weighted_mean_filter(medical_image, kernel_size=5)

# 保存结果
cv2.imwrite('xray_denoised.png', denoised)
```

### 2. 扫描文档去噪

```python
# 加载扫描文档
document = cv2.imread('scan.png', cv2.IMREAD_GRAYSCALE)

# 使用中值滤波去除椒盐噪声
clean_doc = filter_obj.median_filter(document, kernel_size=3)

cv2.imwrite('scan_clean.png', clean_doc)
```

### 3. 照片锐化

```python
# 加载照片
photo = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)

# 拉普拉斯锐化
sharpened = filter_obj.laplacian_sharpen(photo, kernel_type='4', alpha=1.5)

cv2.imwrite('photo_sharp.jpg', sharpened)
```

### 4. 边缘检测

```python
# 加载图像
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# LoG边缘检测
log_result = filter_obj.log_filter(image, sigma=1.5)

# 取绝对值并二值化
edges = np.abs(log_result).astype(np.uint8)
_, binary_edges = cv2.threshold(edges, 10, 255, cv2.THRESH_BINARY)

cv2.imwrite('edges.png', binary_edges)
```

### 5. 组合滤波

```python
# 先去噪，再锐化
denoised = filter_obj.weighted_mean_filter(noisy_image, kernel_size=5)
sharpened = filter_obj.laplacian_sharpen(denoised, alpha=1.0)

cv2.imwrite('enhanced.png', sharpened)
```

## 技术细节

### 均值滤波原理

均值滤波是最简单的线性滤波器，用邻域像素的平均值替代中心像素：

```
g(x,y) = (1/MN) × Σ f(s,t)
```

其中 (s,t) 是 (x,y) 邻域内的像素。

**优点：**
- 简单高效
- 去除高斯噪声效果好

**缺点：**
- 模糊边缘
- 不能去除椒盐噪声

### 中值滤波原理

中值滤波是非线性滤波器，用邻域像素的中值替代中心像素：

```
g(x,y) = median{f(s,t)}
```

**优点：**
- 有效去除椒盐噪声
- 保持边缘

**缺点：**
- 计算复杂度高
- 对高斯噪声效果一般

### 拉普拉斯滤波原理

拉普拉斯算子是二阶微分算子，对图像进行二阶求导：

```
∇²f = ∂²f/∂x² + ∂²f/∂y²
```

**特性：**
- 各向同性（旋转不变）
- 对噪声敏感
- 检测边缘和细节

**锐化原理：**
```
锐化 = 原图 - 拉普拉斯响应
```

### 边界处理

所有滤波器都使用OpenCV的边界处理：
- `BORDER_REFLECT` - 反射边界
- `BORDER_REPLICATE` - 复制边界
- `BORDER_CONSTANT` - 常数边界

## 性能优化

1. **使用OpenCV内置函数**：比纯Python实现快10-100倍
2. **避免循环**：使用向量化操作
3. **合理选择卷积核大小**：3×3通常足够
4. **批处理**：一次处理多张图像

## 常见问题

### Q1: 均值滤波和中值滤波如何选择？
A: 
- **高斯噪声** → 均值滤波或加权均值滤波
- **椒盐噪声** → 中值滤波
- **混合噪声** → 先中值后均值

### Q2: 拉普拉斯锐化后图像过亮或过暗？
A: 调整alpha参数：
- alpha < 1.0：弱锐化
- alpha = 1.0：标准锐化
- alpha > 1.0：强锐化

### Q3: 自适应中值滤波很慢？
A: 自适应中值滤波计算复杂度高，建议：
- 减小max_kernel_size
- 只在必要时使用
- 或使用标准中值滤波

### Q4: 如何选择卷积核大小？
A: 
- **3×3**：标准选择，速度快
- **5×5**：更强的平滑，适合噪声较大的图像
- **7×7及以上**：很强的平滑，可能过度模糊

### Q5: LoG滤波器的sigma如何选择？
A: 
- **sigma = 0.5-1.0**：检测细小边缘
- **sigma = 1.0-2.0**：标准边缘检测
- **sigma > 2.0**：检测粗大边缘

## 扩展功能建议

1. **更多滤波器**
   - 双边滤波
   - 导向滤波
   - 非局部均值滤波

2. **频率域滤波**
   - 理想低通/高通滤波
   - 巴特沃斯滤波
   - 高斯滤波

3. **形态学滤波**
   - 腐蚀/膨胀
   - 开运算/闭运算
   - 形态学梯度

4. **自适应滤波**
   - 维纳滤波
   - 自适应均值滤波

## 许可证

MIT License

## 作者

数字图像处理项目

## 更新日志

### v1.0.0 (2026-06-01)
- 初始版本
- 实现均值、中值、拉普拉斯滤波
- 完整的测试覆盖
- 可视化功能
- 中文文档
