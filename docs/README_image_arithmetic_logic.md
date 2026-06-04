# 图像算术和逻辑操作

## 项目简介

本项目实现了完整的图像算术和逻辑操作功能，包括：

### 算术操作
- **加法**：支持加权加法，自动处理溢出
- **减法**：自动处理下溢保护
- **乘法**：支持缩放因子，自动归一化
- **除法**：自动处理除零保护

### 逻辑操作
- **按位与（AND）**：两图像对应像素按位与
- **按位或（OR）**：两图像对应像素按位或
- **按位非（NOT）**：图像像素按位取反
- **按位异或（XOR）**：两图像对应像素按位异或

## 功能特点

1. **自动边界处理**：所有操作自动处理溢出、下溢和除零情况
2. **多通道支持**：支持灰度图和彩色图像
3. **尺寸自适应**：自动调整不同尺寸图像
4. **可视化展示**：提供完整的可视化功能
5. **完整测试**：包含21个单元测试，覆盖所有功能和边界情况

## 文件结构

```
.
├── image_arithmetic_logic.py          # 主程序文件
├── test_image_arithmetic_logic.py     # 测试文件
├── README_image_arithmetic_logic.md   # 本文档
├── arithmetic_operations.png          # 算术操作演示结果
├── logic_operations.png               # 逻辑操作演示结果
├── test_image1.png                    # 测试图像1
└── test_image2.png                    # 测试图像2
```

## 依赖项

```bash
pip install opencv-python numpy matplotlib pytest
```

## 使用方法

### 1. 基本使用

```python
from image_arithmetic_logic import ImageArithmeticLogic
import cv2

# 创建处理器
processor = ImageArithmeticLogic()

# 加载图像
image1 = cv2.imread('image1.png')
image2 = cv2.imread('image2.png')

# 执行算术操作
add_result = processor.add(image1, image2, 0.5, 0.5)  # 加权加法
subtract_result = processor.subtract(image1, image2)   # 减法
multiply_result = processor.multiply(image1, image2)   # 乘法
divide_result = processor.divide(image1, image2)       # 除法

# 执行逻辑操作
and_result = processor.bitwise_and(image1, image2)     # 按位与
or_result = processor.bitwise_or(image1, image2)       # 按位或
not_result = processor.bitwise_not(image1)             # 按位非
xor_result = processor.bitwise_xor(image1, image2)     # 按位异或

# 保存结果
cv2.imwrite('result.png', add_result)
```

### 2. 可视化所有操作

```python
from image_arithmetic_logic import ImageArithmeticLogic
import cv2

processor = ImageArithmeticLogic()
image1 = cv2.imread('image1.png')
image2 = cv2.imread('image2.png')

# 可视化算术操作
processor.visualize_arithmetic_operations(
    image1, image2, 
    save_path='arithmetic_results.png'
)

# 可视化逻辑操作
processor.visualize_logic_operations(
    image1, image2,
    save_path='logic_results.png'
)
```

### 3. 使用加载功能

```python
processor = ImageArithmeticLogic()

# 加载单张图像
processor.load_images('image1.png')

# 加载两张图像（自动调整尺寸）
processor.load_images('image1.png', 'image2.png')

# 使用加载的图像
result = processor.add(processor.image1, processor.image2)
```

### 4. 运行演示程序

```bash
python image_arithmetic_logic.py
```

这将：
1. 创建两张测试图像
2. 执行所有算术操作并保存结果
3. 执行所有逻辑操作并保存结果

## 运行测试

```bash
# 运行所有测试
python -m pytest test_image_arithmetic_logic.py -v

# 运行特定测试
python -m pytest test_image_arithmetic_logic.py::TestImageArithmeticLogic::test_add_basic -v

# 查看测试覆盖率
python -m pytest test_image_arithmetic_logic.py --cov=image_arithmetic_logic --cov-report=html
```

## API 文档

### ImageArithmeticLogic 类

#### 算术操作方法

##### `add(image1, image2, weight1=1.0, weight2=1.0)`
图像加法操作。

**参数：**
- `image1` (np.ndarray): 第一张图像
- `image2` (np.ndarray): 第二张图像
- `weight1` (float): 第一张图像权重，默认1.0
- `weight2` (float): 第二张图像权重，默认1.0

**返回：**
- `np.ndarray`: 相加后的图像

**示例：**
```python
# 简单相加
result = processor.add(img1, img2)

# 加权平均（图像融合）
result = processor.add(img1, img2, 0.5, 0.5)

# 自定义权重
result = processor.add(img1, img2, 0.7, 0.3)
```

##### `subtract(image1, image2)`
图像减法操作。

**参数：**
- `image1` (np.ndarray): 被减图像
- `image2` (np.ndarray): 减数图像

**返回：**
- `np.ndarray`: 相减后的图像（自动处理下溢）

**应用场景：**
- 背景差分
- 运动检测
- 图像对比

##### `multiply(image1, image2, scale=1.0)`
图像乘法操作。

**参数：**
- `image1` (np.ndarray): 第一张图像
- `image2` (np.ndarray): 第二张图像
- `scale` (float): 缩放因子，默认1.0

**返回：**
- `np.ndarray`: 相乘后的图像

**应用场景：**
- 图像掩膜
- 亮度调整
- 图像融合

##### `divide(image1, image2, scale=255.0)`
图像除法操作。

**参数：**
- `image1` (np.ndarray): 被除图像
- `image2` (np.ndarray): 除数图像
- `scale` (float): 缩放因子，默认255.0

**返回：**
- `np.ndarray`: 相除后的图像（自动处理除零）

**应用场景：**
- 光照校正
- 归一化
- 比率计算

#### 逻辑操作方法

##### `bitwise_and(image1, image2)`
按位与操作。

**参数：**
- `image1` (np.ndarray): 第一张图像
- `image2` (np.ndarray): 第二张图像

**返回：**
- `np.ndarray`: 按位与结果

**应用场景：**
- 图像掩膜
- ROI提取
- 图像合成

##### `bitwise_or(image1, image2)`
按位或操作。

**参数：**
- `image1` (np.ndarray): 第一张图像
- `image2` (np.ndarray): 第二张图像

**返回：**
- `np.ndarray`: 按位或结果

**应用场景：**
- 图像合并
- 区域联合
- 特征组合

##### `bitwise_not(image)`
按位非操作。

**参数：**
- `image` (np.ndarray): 输入图像

**返回：**
- `np.ndarray`: 按位非结果

**应用场景：**
- 图像反转
- 负片效果
- 掩膜反转

##### `bitwise_xor(image1, image2)`
按位异或操作。

**参数：**
- `image1` (np.ndarray): 第一张图像
- `image2` (np.ndarray): 第二张图像

**返回：**
- `np.ndarray`: 按位异或结果

**应用场景：**
- 图像差异检测
- 加密/解密
- 变化检测

#### 可视化方法

##### `visualize_arithmetic_operations(image1, image2, save_path=None)`
可视化所有算术操作。

**参数：**
- `image1` (np.ndarray): 第一张图像
- `image2` (np.ndarray): 第二张图像
- `save_path` (str, optional): 保存路径

##### `visualize_logic_operations(image1, image2, save_path=None)`
可视化所有逻辑操作。

**参数：**
- `image1` (np.ndarray): 第一张图像
- `image2` (np.ndarray): 第二张图像
- `save_path` (str, optional): 保存路径

## 实际应用示例

### 1. 图像融合（加权加法）

```python
# 将两张图像融合，创建过渡效果
foreground = cv2.imread('foreground.png')
background = cv2.imread('background.png')

# 70%前景 + 30%背景
blended = processor.add(foreground, background, 0.7, 0.3)
cv2.imwrite('blended.png', blended)
```

### 2. 背景差分（减法）

```python
# 检测运动物体
background = cv2.imread('background.png')
current_frame = cv2.imread('current_frame.png')

# 提取前景
foreground = processor.subtract(current_frame, background)
cv2.imwrite('foreground.png', foreground)
```

### 3. 图像掩膜（乘法/与操作）

```python
# 使用掩膜提取感兴趣区域
image = cv2.imread('image.png')
mask = cv2.imread('mask.png')

# 方法1：使用乘法
masked1 = processor.multiply(image, mask, 1.0/255.0)

# 方法2：使用按位与
masked2 = processor.bitwise_and(image, mask)

cv2.imwrite('masked.png', masked1)
```

### 4. 变化检测（异或）

```python
# 检测两张图像的差异
image1 = cv2.imread('before.png')
image2 = cv2.imread('after.png')

# 提取变化区域
changes = processor.bitwise_xor(image1, image2)
cv2.imwrite('changes.png', changes)
```

### 5. 图像反转（非操作）

```python
# 创建负片效果
image = cv2.imread('image.png')
negative = processor.bitwise_not(image)
cv2.imwrite('negative.png', negative)
```

## 测试覆盖

项目包含21个单元测试，覆盖：

1. **算术操作测试**（8个）
   - 基本加法和加权加法
   - 减法和下溢保护
   - 乘法和溢出保护
   - 除法和除零保护

2. **逻辑操作测试**（4个）
   - 按位与、或、非、异或

3. **彩色图像测试**（2个）
   - 彩色图像算术操作
   - 彩色图像逻辑操作

4. **边界情况测试**（3个）
   - 全零图像
   - 全255图像
   - 相同图像操作

5. **图像加载测试**（4个）
   - 单张图像加载
   - 两张图像加载
   - 不存在图像处理
   - 不同尺寸图像自适应

**测试结果：** 21/21 通过 ✅

## 技术细节

### 溢出和下溢处理

- **加法溢出**：使用 `cv2.addWeighted()` 自动截断到255
- **减法下溢**：使用 `cv2.subtract()` 自动截断到0
- **乘法溢出**：转换为float32计算后使用 `np.clip()` 截断
- **除零保护**：将除数中的0替换为1

### 数据类型处理

所有操作保持 `uint8` 数据类型（0-255范围），确保：
- 内存效率
- 与OpenCV函数兼容
- 可直接保存和显示

### 性能优化

- 使用OpenCV内置函数（C++实现，速度快）
- 避免不必要的数据类型转换
- 支持原地操作（减少内存分配）

## 常见问题

### Q1: 为什么加法结果看起来很暗？
A: 如果直接相加（weight1=1.0, weight2=1.0），很容易溢出到255。建议使用加权加法，如 `add(img1, img2, 0.5, 0.5)`。

### Q2: 除法结果为什么全黑？
A: 默认scale=255.0适用于归一化场景。如果需要直接除法，使用 `divide(img1, img2, scale=1.0)`。

### Q3: 逻辑操作对彩色图像有效吗？
A: 有效。逻辑操作对每个通道独立进行。

### Q4: 如何处理不同尺寸的图像？
A: 使用 `load_images()` 方法会自动调整第二张图像尺寸，或手动使用 `cv2.resize()`。

### Q5: 中文标题显示为方框怎么办？
A: 代码已配置中文字体支持。如果仍有问题，安装中文字体或修改 `plt.rcParams['font.sans-serif']`。

## 扩展功能建议

1. **高级融合算法**
   - 泊松融合
   - 多分辨率融合
   - Alpha混合

2. **更多算术操作**
   - 平方根
   - 对数运算
   - 指数运算

3. **批处理支持**
   - 处理图像序列
   - 视频帧操作

4. **GPU加速**
   - 使用CUDA
   - OpenCL支持

## 许可证

MIT License

## 作者

数字图像处理项目

## 更新日志

### v1.0.0 (2026-06-01)
- 初始版本
- 实现所有基本算术和逻辑操作
- 完整的测试覆盖
- 可视化功能
- 中文文档
