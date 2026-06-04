# 图像压缩编码

## 项目简介

本项目实现了四种经典的无损压缩编码算法：

1. **哈夫曼编码** (Huffman Coding) - 基于频率的变长编码
2. **算术编码** (Arithmetic Coding) - 基于概率的区间编码
3. **LZW编码** (Lempel-Ziv-Welch) - 字典压缩算法
4. **位平面编码** (Bit-Plane Coding) - 结合游程编码

## 功能特点

1. ✅ **4种压缩算法**：覆盖不同的压缩策略
2. ✅ **完整编码/解码**：支持压缩和解压缩
3. ✅ **无损压缩**：完美重建原始图像
4. ✅ **压缩比计算**：自动评估压缩效果
5. ✅ **算法比较**：一键比较所有算法
6. ✅ **完整测试**：20个单元测试

## 快速开始

```python
from image_compression import HuffmanCoding, LZWCoding, BitPlaneCoding
import numpy as np

image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

# 哈夫曼编码
huffman = HuffmanCoding()
encoded, metadata = huffman.encode(image)
decoded = huffman.decode(encoded, metadata)
ratio = huffman.calculate_compression_ratio(image, encoded)

# LZW编码
lzw = LZWCoding()
encoded_lzw, meta_lzw = lzw.encode(image)
decoded_lzw = lzw.decode(encoded_lzw, meta_lzw)

# 位平面编码
bitplane = BitPlaneCoding()
encoded_bp, meta_bp = bitplane.encode(image)
decoded_bp = bitplane.decode(encoded_bp, meta_bp)

# 比较所有算法
from image_compression import CompressionComparison
comparison = CompressionComparison()
results = comparison.compare_all(image)
```

## 算法特性对比

| 算法 | 压缩策略 | 适用场景 | 复杂度 |
|------|---------|---------|--------|
| **哈夫曼** | 频率统计 | 符号分布不均匀 | O(n log n) |
| **算术** | 概率区间 | 高压缩比需求 | O(n) |
| **LZW** | 字典匹配 | 重复模式多 | O(n) |
| **位平面** | 游程编码 | 二值/简单图像 | O(n) |

## API 文档

### 哈夫曼编码

```python
huffman = HuffmanCoding()

# 编码
encoded, metadata = huffman.encode(image)
# - encoded: 二进制字符串
# - metadata: 包含编码表和图像信息

# 解码
decoded = huffman.decode(encoded, metadata)

# 压缩比
ratio = huffman.calculate_compression_ratio(image, encoded)
```

**原理：**
1. 统计每个像素值的频率
2. 构建哈夫曼树
3. 生成变长编码（高频→短码，低频→长码）

### LZW编码

```python
lzw = LZWCoding()

# 编码
encoded, metadata = lzw.encode(image)
# - encoded: 码字列表

# 解码
decoded = lzw.decode(encoded, metadata)
```

**原理：**
1. 初始化字典（0-255）
2. 查找最长匹配
3. 动态添加新模式到字典

### 位平面编码

```python
bitplane = BitPlaneCoding()

# 编码（仅支持灰度图像）
encoded, metadata = bitplane.encode(image)
# - encoded: 8个位平面的游程编码

# 解码
decoded = bitplane.decode(encoded, metadata)
```

**原理：**
1. 分解8个位平面（每个像素的每一位）
2. 对每个位平面进行游程编码
3. 高位平面通常更稀疏，压缩效果更好

## 算法详解

### 哈夫曼编码

**优点：**
- 简单易实现
- 压缩速度快
- 符号分布不均时效果好

**缺点：**
- 需要两遍扫描（统计+编码）
- 需要传输编码表
- 随机数据压缩效果差

### LZW编码

**优点：**
- 自适应字典，无需传输
- 对重复模式效果好
- 广泛应用（GIF、TIFF）

**缺点：**
- 字典可能很大
- 首次出现的模式无法压缩

### 位平面编码

**优点：**
- 高位平面压缩效果好
- 可以渐进传输（从高位到低位）
- 适合二值图像

**缺点：**
- 需要分解和重组
- 对复杂图像效果一般

## 实际应用

### 1. 文档图像压缩
```python
# 扫描的二值文档
document = cv2.imread('document.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(document, 127, 255, cv2.THRESH_BINARY)

# 使用位平面编码（对二值图像效果好）
bitplane = BitPlaneCoding()
encoded, metadata = bitplane.encode(binary)
```

### 2. 重复模式图像
```python
# 有重复纹理的图像
pattern_image = cv2.imread('pattern.png', cv2.IMREAD_GRAYSCALE)

# 使用LZW编码（对重复模式效果好）
lzw = LZWCoding()
encoded, metadata = lzw.encode(pattern_image)
```

### 3. 通用图像压缩
```python
# 一般图像
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# 使用哈夫曼编码（通用性好）
huffman = HuffmanCoding()
encoded, metadata = huffman.encode(image)
```

## 压缩效果分析

### 典型压缩比

| 图像类型 | 哈夫曼 | LZW | 位平面 |
|---------|-------|-----|-------|
| 随机噪声 | 1.0:1 | 1.0:1 | <1:1 |
| 自然图像 | 1.2:1 | 1.5:1 | 1.0:1 |
| 重复纹理 | 1.5:1 | 3.0:1 | 1.2:1 |
| 二值图像 | 2.0:1 | 2.5:1 | 5.0:1 ⭐ |

**注意：** 位平面编码对二值图像效果最好

## 技术原理

### 哈夫曼树构建

```
频率: A=5, B=9, C=12, D=13, E=16, F=45

构建过程:
1. 创建叶节点，按频率排序
2. 合并两个最小的节点
3. 重复直到只剩一个根节点

编码结果:
F → 0
C → 100
D → 101
A → 1100
B → 1101
E → 111
```

### 游程编码

```
原始: 0 0 0 0 1 1 0 0 0 1
编码: (0,4) (1,2) (0,3) (1,1)

压缩比 = 10 / (4×2) = 1.25:1
```

## 常见问题

### Q1: 为什么随机图像压缩比接近1:1？

A: 随机图像没有冗余信息，熵接近最大值，无法有效压缩。

### Q2: 哈夫曼 vs LZW 如何选择？

A:
- **哈夫曼**：符号分布不均匀
- **LZW**：有重复模式

### Q3: 位平面编码适合什么图像？

A: 二值图像或低位深度图像（高位平面稀疏）

### Q4: 为什么需要传输元数据？

A: 解码器需要编码表、图像尺寸等信息才能正确解码。

## 测试结果

```
============================== 20 passed in 0.41s ==============================
```

**测试覆盖：**
- 编码/解码正确性
- 压缩比计算
- 边界情况（空图像、单像素、大图像）
- 特殊图像（二值、均匀）

## 许可证

MIT License

## 更新日志

### v1.0.0 (2026-06-03)
- ✅ 实现哈夫曼编码
- ✅ 实现算术编码
- ✅ 实现LZW编码
- ✅ 实现位平面编码（游程编码）
- ✅ 20个单元测试全部通过
- ✅ 算法比较功能
- ✅ 中文文档
