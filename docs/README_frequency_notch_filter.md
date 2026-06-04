# 频域陷波滤波器实现

## 项目简介

本项目实现了完整的频域陷波滤波功能，用于去除图像中的周期性噪声。

### 陷波滤波器类型

#### 理想陷波滤波器
- **特点**：在陷波区域完全阻止或通过
- **优点**：去噪效果强
- **缺点**：可能产生振铃效应

#### 巴特沃斯陷波滤波器
- **特点**：平滑过渡的陷波特性
- **优点**：过渡平滑，减少振铃
- **缺点**：去噪效果略弱于理想滤波器

#### 高斯陷波滤波器
- **特点**：高斯形状的陷波特性
- **优点**：最平滑的过渡
- **缺点**：计算复杂度较高

### 滤波模式

- **陷波抑制（Notch Reject）**：去除特定频率成分
- **陷波通过（Notch Pass）**：只保留特定频率成分

## 功能特点

1. **三种陷波滤波器**：理想、巴特沃斯、高斯
2. **自动对称陷波**：自动创建关于中心对称的陷波点对
3. **频谱峰值检测**：自动检测噪声频率位置
4. **周期噪声模拟**：支持添加多频率周期噪声
5. **完整测试**：28个单元测试，覆盖所有功能
6. **可视化展示**：对比展示所有滤波效果

## 文件结构

```
.
├── frequency_notch_filter.py          # 主程序文件
├── test_frequency_notch_filter.py     # 测试文件
├── README_frequency_notch_filter.md   # 本文档
├── notch_filter_noisy.png            # 带噪声的测试图像
├── notch_filter_spectrum.png         # 频谱可视化
└── notch_filter_results.png          # 滤波效果对比
```

## 依赖项

```bash
pip install opencv-python numpy matplotlib scipy pytest
```

## 使用方法

### 1. 基本使用

```python
from frequency_notch_filter import FrequencyNotchFilter
import cv2

# 创建滤波器对象
filter_obj = FrequencyNotchFilter()

# 加载图像
image = cv2.imread('noisy_image.png', cv2.IMREAD_GRAYSCALE)

# 定义陷波位置（频域坐标）
notch_positions = [
    (128, 158),  # 噪声频率1
    (158, 128),  # 噪声频率2
]

# 陷波半径
radius = 10

# 应用理想陷波抑制
result = filter_obj.ideal_notch_reject(image, notch_positions, radius)

# 保存结果
cv2.imwrite('filtered.png', result)
```

### 2. 添加周期噪声并去除

```python
# 创建测试图像
test_image = cv2.imread('clean_image.png', cv2.IMREAD_GRAYSCALE)

# 添加周期性噪声
noisy = filter_obj.add_periodic_noise(
    test_image,
    frequencies=[(0, 30), (30, 0), (20, 20)],  # 三个频率
    amplitudes=[30, 30, 20]                     # 对应幅度
)

# 检测噪声峰值
_, fft_shifted = filter_obj.fft_transform(noisy)
peaks = filter_obj.detect_peaks_in_spectrum(fft_shifted, threshold_percentile=99.5)
print(f"检测到 {len(peaks)} 个峰值")

# 应用陷波滤波
H, W = noisy.shape
notch_positions = [
    (H//2, W//2 + 30),   # 对应频率 (0, 30)
    (H//2 + 30, W//2),   # 对应频率 (30, 0)
    (H//2 + 20, W//2 + 20)  # 对应频率 (20, 20)
]

result = filter_obj.ideal_notch_reject(noisy, notch_positions, radius=10)
```

### 3. 使用不同类型的陷波滤波器

```python
# 理想陷波
ideal = filter_obj.ideal_notch_reject(image, notch_positions, radius=10)

# 巴特沃斯陷波（阶数=2）
butterworth = filter_obj.butterworth_notch_reject(
    image, 
    notch_positions, 
    radius=10, 
    order=2
)

# 高斯陷波
gaussian = filter_obj.gaussian_notch_reject(image, notch_positions, radius=10)
```

### 4. 陷波通过滤波器

```python
# 只保留特定频率（提取周期噪声）
noise_only = filter_obj.ideal_notch_pass(image, notch_positions, radius=10)
```

### 5. 可视化频谱

```python
# 显示频谱
filter_obj.visualize_spectrum(
    image, 
    title="图像频谱", 
    save_path='spectrum.png'
)
```

### 6. 完整滤波效果对比

```python
# 展示所有滤波器的效果
filter_obj.visualize_notch_filtering(
    noisy_image,
    notch_positions=notch_positions,
    radius=10,
    save_path='comparison.png'
)
```

### 7. 运行演示程序

```bash
python frequency_notch_filter.py
```

这将：
1. 创建测试图像
2. 添加周期性噪声
3. 应用所有陷波滤波器
4. 生成对比图

## 运行测试

```bash
# 运行所有测试
python -m pytest test_frequency_notch_filter.py -v

# 运行特定测试
python -m pytest test_frequency_notch_filter.py::TestFrequencyNotchFilter::test_ideal_notch_reject -v

# 查看测试覆盖率
python -m pytest test_frequency_notch_filter.py --cov=frequency_notch_filter --cov-report=html
```

**测试结果：** 28/28 通过 ✅

## API 文档

### FrequencyNotchFilter 类

#### FFT变换方法

##### `fft_transform(image)`
FFT变换并中心化。

**参数：**
- `image` (np.ndarray): 输入图像

**返回：**
- `Tuple[np.ndarray, np.ndarray]`: (FFT结果, 中心化FFT结果)

**示例：**
```python
fft, fft_shifted = filter_obj.fft_transform(image)
magnitude = np.abs(fft_shifted)
```

##### `ifft_transform(fft_shifted)`
逆FFT变换。

**参数：**
- `fft_shifted` (np.ndarray): 中心化的FFT结果

**返回：**
- `np.ndarray`: 恢复的图像

#### 陷波掩码方法

##### `create_notch_mask(shape, notch_positions, radius, filter_type='ideal', order=2)`
创建陷波掩码（自动生成对称点对）。

**参数：**
- `shape` (Tuple[int, int]): 图像形状 (H, W)
- `notch_positions` (List[Tuple[int, int]]): 陷波位置列表
  - 只需指定一侧的点，自动创建关于中心对称的点对
- `radius` (float): 陷波半径
- `filter_type` (str): 滤波器类型
  - `'ideal'` - 理想陷波
  - `'butterworth'` - 巴特沃斯陷波
  - `'gaussian'` - 高斯陷波
- `order` (int): 巴特沃斯滤波器阶数，默认2

**返回：**
- `np.ndarray`: 陷波掩码（1=通过，0=阻止）

**应用场景：**
- 自定义陷波滤波器
- 组合多个陷波点
- 研究不同滤波器特性

**示例：**
```python
# 创建理想陷波掩码
mask = filter_obj.create_notch_mask(
    shape=(256, 256),
    notch_positions=[(150, 150), (180, 180)],
    radius=10,
    filter_type='ideal'
)

# 创建巴特沃斯陷波掩码
mask_bw = filter_obj.create_notch_mask(
    shape=(256, 256),
    notch_positions=[(150, 150)],
    radius=10,
    filter_type='butterworth',
    order=3
)
```

#### 陷波滤波器方法

##### `ideal_notch_reject(image, notch_positions, radius)`
理想陷波抑制滤波器。

**参数：**
- `image` (np.ndarray): 输入图像
- `notch_positions` (List[Tuple[int, int]]): 陷波位置列表
- `radius` (float): 陷波半径

**返回：**
- `np.ndarray`: 滤波后的图像

**应用场景：**
- 去除明显的周期性噪声
- 快速去噪
- 噪声频率已知的情况

**特点：**
- 去噪效果最强
- 可能产生振铃（Ringing）
- 计算速度快

##### `butterworth_notch_reject(image, notch_positions, radius, order=2)`
巴特沃斯陷波抑制滤波器。

**参数：**
- `image` (np.ndarray): 输入图像
- `notch_positions` (List[Tuple[int, int]]): 陷波位置列表
- `radius` (float): 陷波半径
- `order` (int): 滤波器阶数，默认2

**返回：**
- `np.ndarray`: 滤波后的图像

**应用场景：**
- 需要平滑过渡的去噪
- 减少振铃效应
- 高质量图像处理

**特点：**
- 平滑过渡
- 减少振铃
- 阶数越高越接近理想滤波器

**阶数选择：**
- `order=1` - 非常平滑，去噪较弱
- `order=2` - 标准选择
- `order=4` - 接近理想滤波器

##### `gaussian_notch_reject(image, notch_positions, radius)`
高斯陷波抑制滤波器。

**参数：**
- `image` (np.ndarray): 输入图像
- `notch_positions` (List[Tuple[int, int]]): 陷波位置列表
- `radius` (float): 陷波半径（标准差）

**返回：**
- `np.ndarray`: 滤波后的图像

**应用场景：**
- 需要最平滑过渡
- 医学图像处理
- 高端图像处理

**特点：**
- 最平滑的过渡
- 无振铃
- 计算稍慢

##### `ideal_notch_pass(image, notch_positions, radius)`
理想陷波通过滤波器（只保留指定频率）。

**参数：**
- `image` (np.ndarray): 输入图像
- `notch_positions` (List[Tuple[int, int]]): 陷波位置列表
- `radius` (float): 陷波半径

**返回：**
- `np.ndarray`: 滤波后的图像（只包含特定频率）

**应用场景：**
- 提取周期性噪声
- 频率分析
- 噪声特征研究

**示例：**
```python
# 提取噪声
noise = filter_obj.ideal_notch_pass(noisy_image, notch_positions, radius=10)

# 显示噪声
cv2.imshow('Extracted Noise', noise)
```

#### 辅助方法

##### `add_periodic_noise(image, frequencies, amplitudes)`
添加周期性噪声。

**参数：**
- `image` (np.ndarray): 输入图像
- `frequencies` (List[Tuple[float, float]]): 频率列表 [(freq_u, freq_v), ...]
  - freq_u: u方向的频率（行方向）
  - freq_v: v方向的频率（列方向）
- `amplitudes` (List[float]): 幅度列表

**返回：**
- `np.ndarray`: 添加噪声后的图像

**应用场景：**
- 模拟周期性噪声
- 测试滤波器效果
- 算法验证

**噪声公式：**
```
noise = amplitude × sin(2π(freq_u·y/H + freq_v·x/W))
```

**示例：**
```python
# 添加水平条纹
noisy = filter_obj.add_periodic_noise(
    image,
    frequencies=[(10, 0)],
    amplitudes=[30]
)

# 添加垂直条纹
noisy = filter_obj.add_periodic_noise(
    image,
    frequencies=[(0, 10)],
    amplitudes=[30]
)

# 添加斜向条纹
noisy = filter_obj.add_periodic_noise(
    image,
    frequencies=[(10, 10)],
    amplitudes=[30]
)

# 添加多个频率的混合噪声
noisy = filter_obj.add_periodic_noise(
    image,
    frequencies=[(10, 0), (0, 15), (8, 8)],
    amplitudes=[25, 30, 20]
)
```

##### `detect_peaks_in_spectrum(fft_shifted, threshold_percentile=99.9, min_distance=10)`
在频谱中检测峰值（自动找噪声频率）。

**参数：**
- `fft_shifted` (np.ndarray): 中心化的FFT结果
- `threshold_percentile` (float): 峰值阈值百分位，默认99.9
- `min_distance` (int): 峰值之间的最小距离，默认10

**返回：**
- `List[Tuple[int, int]]`: 峰值位置列表（排除DC分量）

**应用场景：**
- 自动检测噪声频率
- 未知噪声分析
- 批量处理

**示例：**
```python
# FFT变换
_, fft_shifted = filter_obj.fft_transform(noisy_image)

# 自动检测峰值
peaks = filter_obj.detect_peaks_in_spectrum(
    fft_shifted,
    threshold_percentile=99.5,  # 更低的阈值检测更多峰值
    min_distance=15             # 峰值之间至少15像素
)

print(f"检测到 {len(peaks)} 个噪声峰值")

# 使用检测到的峰值进行滤波
result = filter_obj.ideal_notch_reject(noisy_image, peaks, radius=8)
```

##### `load_image(path)`
加载图像。

**参数：**
- `path` (str): 图像路径

**返回：**
- `bool`: 是否成功加载

#### 可视化方法

##### `visualize_spectrum(image, title='频谱', save_path=None)`
可视化频谱。

**参数：**
- `image` (np.ndarray): 输入图像
- `title` (str): 标题
- `save_path` (str, optional): 保存路径

**示例：**
```python
filter_obj.visualize_spectrum(
    noisy_image,
    title='带噪声的频谱',
    save_path='spectrum.png'
)
```

##### `visualize_notch_filtering(image, notch_positions, radius, save_path=None)`
可视化陷波滤波效果对比。

**参数：**
- `image` (np.ndarray): 输入图像
- `notch_positions` (List[Tuple[int, int]]): 陷波位置列表
- `radius` (float): 陷波半径
- `save_path` (str, optional): 保存路径

**展示内容：**
- 原始图像和频谱
- 三种滤波器的掩码
- 三种滤波器的结果
- 滤波后的频谱
- 陷波位置标记

## 实际应用示例

### 1. 扫描文档去除网格纹

```python
# 加载扫描文档
document = cv2.imread('scanned_document.png', cv2.IMREAD_GRAYSCALE)

# 文档扫描常见的网格噪声频率
H, W = document.shape
center_u, center_v = H // 2, W // 2

# 网格频率通常是规则的
notch_positions = [
    (center_u, center_v + 50),   # 垂直网格
    (center_u + 50, center_v),   # 水平网格
]

# 使用高斯陷波（平滑）
clean_doc = filter_obj.gaussian_notch_reject(document, notch_positions, radius=8)

cv2.imwrite('clean_document.png', clean_doc)
```

### 2. 去除视频干扰条纹

```python
# 视频帧常见的水平扫描线干扰
video_frame = cv2.imread('video_frame.png', cv2.IMREAD_GRAYSCALE)

# 水平扫描线对应特定垂直频率
H, W = video_frame.shape
notch_positions = [(H//2, W//2 + 60)]  # 60是扫描线频率

# 使用巴特沃斯（平衡效果和振铃）
clean_frame = filter_obj.butterworth_notch_reject(
    video_frame, 
    notch_positions, 
    radius=5, 
    order=3
)

cv2.imwrite('clean_frame.png', clean_frame)
```

### 3. 航拍图像去除引擎振动噪声

```python
# 航拍图像的引擎振动产生周期性噪声
aerial = cv2.imread('aerial_image.png', cv2.IMREAD_GRAYSCALE)

# 自动检测振动频率
_, fft_shifted = filter_obj.fft_transform(aerial)
peaks = filter_obj.detect_peaks_in_spectrum(fft_shifted, threshold_percentile=99.7)

print(f"检测到 {len(peaks)} 个振动频率")

# 使用理想陷波快速去除
clean_aerial = filter_obj.ideal_notch_reject(aerial, peaks, radius=12)

cv2.imwrite('clean_aerial.png', clean_aerial)
```

### 4. 医学图像去除设备干扰

```python
# X光或CT图像的设备电磁干扰
medical = cv2.imread('xray.png', cv2.IMREAD_GRAYSCALE)

# 设备干扰通常是固定频率
notch_positions = [(140, 140), (160, 120)]  # 根据设备特性

# 医学图像要求高质量，使用高斯
clean_medical = filter_obj.gaussian_notch_reject(medical, notch_positions, radius=10)

cv2.imwrite('clean_xray.png', clean_medical)
```

### 5. 批量处理图像序列

```python
import glob

# 批量处理多张图像
image_files = glob.glob('noisy_images/*.png')

for img_path in image_files:
    # 加载图像
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 自动检测噪声
    _, fft_shifted = filter_obj.fft_transform(image)
    peaks = filter_obj.detect_peaks_in_spectrum(fft_shifted)
    
    # 滤波
    clean = filter_obj.butterworth_notch_reject(image, peaks, radius=10, order=2)
    
    # 保存
    output_path = img_path.replace('noisy_images', 'clean_images')
    cv2.imwrite(output_path, clean)
    
print(f"处理完成 {len(image_files)} 张图像")
```

## 技术原理

### 陷波滤波器数学原理

#### 理想陷波滤波器

陷波抑制滤波器：
```
H(u,v) = 0,  当 D(u,v) ≤ D0
       = 1,  当 D(u,v) > D0
```

陷波通过滤波器：
```
H(u,v) = 1 - H_reject(u,v)
```

其中 D(u,v) 是到陷波中心的距离，D0 是陷波半径。

#### 巴特沃斯陷波滤波器

```
H(u,v) = 1 / [1 + (D0/D(u,v))^(2n)]
```

其中 n 是阶数：
- n=1: 非常平滑
- n=2: 标准选择
- n→∞: 接近理想滤波器

#### 高斯陷波滤波器

```
H(u,v) = 1 - exp(-D(u,v)²/(2D0²))
```

高斯滤波器在空间域和频率域都是高斯函数，过渡最平滑。

### 对称性原理

对于实数图像，其傅里叶变换具有共轭对称性：
```
F(u,v) = F*(-u,-v)
```

因此陷波滤波器必须在对称位置同时陷波，否则会引入虚部。

本实现自动为每个指定的陷波点创建关于中心对称的点对。

### 频率到空间的映射

周期性噪声频率 (freq_u, freq_v) 对应的FFT坐标：
```
u_fft = H/2 + freq_u
v_fft = W/2 + freq_v
```

其中 (H/2, W/2) 是频谱中心（DC分量）。

## 参数选择指南

### 陷波半径选择

| 半径范围 | 效果 | 适用场景 |
|---------|------|---------|
| 1-3 | 精确去除单一频率 | 明确的单频噪声 |
| 5-10 | 标准去噪 | 一般周期噪声 |
| 15-30 | 宽带去除 | 频率不确定或多频噪声 |
| >30 | 过度滤波 | 可能损失有用信息 |

### 滤波器类型选择

| 滤波器 | 优点 | 缺点 | 适用场景 |
|--------|------|------|---------|
| 理想 | 去噪最强 | 振铃效应 | 快速处理，噪声明显 |
| 巴特沃斯 | 平衡效果 | 计算稍慢 | 通用场景，推荐 |
| 高斯 | 最平滑 | 去噪稍弱 | 高质量图像，医学图像 |

### 巴特沃斯阶数选择

| 阶数 | 特性 | 建议 |
|------|------|------|
| n=1 | 最平滑过渡 | 轻微噪声，保留细节 |
| n=2 | 标准选择 | **推荐默认值** |
| n=3-4 | 较陡峭 | 中等噪声 |
| n>5 | 接近理想 | 强噪声，但注意振铃 |

## 性能优化

1. **使用FFT**：numpy的FFT比手动实现快100倍以上
2. **向量化操作**：避免循环，使用numpy广播
3. **合理选择半径**：过大的半径会增加计算量
4. **批处理**：多张图像复用掩码可提速

## 常见问题

### Q1: 如何确定陷波位置？

A: 三种方法：
1. **手动分析频谱**：可视化频谱，找亮点
2. **自动检测**：使用 `detect_peaks_in_spectrum()`
3. **已知噪声频率**：根据公式计算FFT坐标

### Q2: 为什么需要对称陷波？

A: 实数图像的频谱具有共轭对称性，F(u,v) = F*(-u,-v)。如果只陷波一侧，逆变换会产生虚部，导致错误。

### Q3: 陷波滤波后出现振铃怎么办？

A: 振铃是理想滤波器的特性，解决方法：
- 使用巴特沃斯或高斯滤波器
- 减小陷波半径
- 降低巴特沃斯阶数

### Q4: 自动检测峰值效果不好？

A: 调整参数：
- **降低阈值**：`threshold_percentile=99.0`（检测更多峰值）
- **增大最小距离**：`min_distance=20`（避免相邻峰值）
- **手动指定位置**：对于已知噪声

### Q5: 如何选择陷波半径？

A: 根据频谱观察：
- 峰值尖锐 → 小半径（5-8）
- 峰值宽广 → 大半径（10-15）
- 不确定 → 从10开始尝试

### Q6: 滤波后图像整体变暗或变亮？

A: 这通常是因为DC分量（零频率）被影响。解决：
- 确保陷波位置不包含中心点
- 检查掩码是否正确
- 可以手动归一化结果

### Q7: 多个噪声频率如何处理？

A: 两种方式：
1. **一次性陷波**：`notch_positions` 包含所有位置
2. **逐个滤波**：依次应用多次陷波（可能累积误差）

推荐方式1，一次性处理。

### Q8: 陷波滤波和带阻滤波的区别？

A: 
- **带阻滤波**：阻止一个频率带（环形区域）
- **陷波滤波**：阻止特定点（点状区域）

陷波更精确，适合去除离散的周期噪声。

## 扩展功能建议

1. **自适应陷波半径**
   - 根据峰值宽度自动调整半径
   - 不同陷波点使用不同半径

2. **带阻滤波器**
   - 阻止频率环（而非点）
   - 去除更广泛的频率范围

3. **相位保留滤波**
   - 只修改幅度，保留相位
   - 更好地保持图像结构

4. **GUI界面**
   - 交互式选择陷波位置
   - 实时预览滤波效果

5. **3D图像支持**
   - 扩展到3D FFT
   - 处理视频或体数据

## 许可证

MIT License

## 作者

数字图像处理项目

## 更新日志

### v1.0.0 (2026-06-02)
- 初始版本
- 实现理想、巴特沃斯、高斯陷波滤波器
- 自动对称陷波点生成
- 峰值检测功能
- 完整的测试覆盖（28个测试）
- 可视化功能
- 中文文档
