# DIP - Digital Image Processing

数字图像处理项目，包含MATLAB经典算法实现和Python细胞分割系统。

---

## 📁 项目结构

### 1. MATLAB 空间域与频率域处理

经典图像处理算法的手动实现。

| 文件 | 内容 |
|------|------|
| `my_median.m` | 手动实现中值滤波（不使用 `medfilt2`） |
| `my_histeq.m` | 手动实现直方图均衡化（不使用 `histeq`） |
| `my_laplacian.m` | 拉普拉斯锐化增强 |
| `fft_demo.m` | FFT 频域分析演示（旋转/平移/尺度/直流/低中高频编辑） |

### 2. Python 细胞分割系统

基于图像处理和深度学习的细胞分割工具，提供命令行和Web界面。

| 文件名 | 说明 | 类型 |
|--------|------|------|
| `app.py` | 传统方法Web界面 | Web应用 |
| `app_cellpose.py` | Cellpose深度学习Web界面 | Web应用 |
| `cell_segmentation_improved.py` | 传统方法命令行版本 | 命令行工具 |
| `cell_segmentation_cellpose.py` | Cellpose命令行版本 | 命令行工具 |
| `requirements.txt` | Python依赖包列表 | 配置文件 |
| `Picture1.png` | 示例细胞图像 | 测试数据 |

---

## 🚀 快速开始

### MATLAB 部分

#### 中值滤波
```matlab
out = my_median(img);          % 默认 3×3 窗口
out = my_median(img, 5);       % 5×5 窗口（去噪更强）
```

#### 直方图均衡化
```matlab
out = my_histeq(img);
```

#### 拉普拉斯锐化
```matlab
out = my_laplacian(img);       % 默认 k=1.0
out = my_laplacian(img, 1.5);  % 增强更强
```

#### FFT 频域分析
```matlab
im = im2double(rgb2gray(imread('your_image.jpg')));
demo_rotation(im, 45);     % 旋转性质
demo_translation(im);      % 平移性质
demo_scale(im);            % 尺度变换
demo_dc(im);               % 直流分量
demo_freq_edit(im);        % 低/中/高频编辑
```

### Python 细胞分割部分

#### 1. 安装依赖
```bash
pip install -r requirements.txt
```

#### 2. 运行Web版本

**传统方法版本：**
```bash
streamlit run app.py
```

**Cellpose深度学习版本：**
```bash
streamlit run app_cellpose.py
```

#### 3. 运行命令行版本

**传统方法版本：**
```bash
python cell_segmentation_improved.py Picture1.png
```

**Cellpose深度学习版本：**
```bash
python cell_segmentation_cellpose.py Picture1.png
```

---

## 📖 详细说明

### MATLAB 算法原理

#### 1. 中值滤波
去除椒盐噪声，保留边缘。对每个像素取其邻域窗口内像素值的中位数代替原值。

**关键步骤：**
1. `padarray(..., 'replicate')` — 边缘复制填充，避免黑边
2. 双重循环滑动窗口，`median(patch(:))` 取中位数
3. 首尾 `double ↔ uint8` 转换防止溢出

#### 2. 直方图均衡化
增强对比度，使暗图变亮、细节更清晰。将像素值的分布从集中拉伸为均匀分布。

**核心公式：**
$$s = \left\lfloor \frac{CDF(r) - CDF_{min}}{N - CDF_{min}} \times 255 + 0.5 \right\rfloor$$

> ⚠️ 彩色图不能直接均衡化（会色偏），需先转 HSV，只对 V 通道操作。

#### 3. 拉普拉斯锐化
增强边缘，使图像更清晰。用拉普拉斯算子提取边缘/细节，叠加回原图。

**卷积核：**
```
 0  -1   0
-1   4  -1
 0  -1   0
```

#### 4. FFT 频域分析

**五大性质总结：**

| 性质 | 空间域操作 | 频域变化 |
|------|-----------|---------|
| 旋转 | 旋转 θ° | 幅度谱同步旋转 θ° |
| 平移 | 循环移位 | 幅度不变，相位线性变化 |
| 尺度 | 放大 | 频域压缩（低频集中） |
| 尺度 | 缩小 | 频域扩展（高频增多） |
| 直流 | 均值亮度 | F(0,0) = 均值 × 像素总数 |

**频率分量视觉效果：**

| 频段 | 范围 | 去掉效果 | 只保留效果 |
|------|------|---------|-----------|
| 低频 | <8% 半径 | 亮度消失，只剩边缘 | 模糊轮廓 |
| 中频 | 8%~25% | 纹理减弱 | 主要纹理结构 |
| 高频 | >25% | 图像模糊（低通滤波） | 只剩边缘轮廓线 |

### Python 细胞分割系统

#### 功能特点
- **传统方法**：Otsu阈值 + 分水岭算法
- **深度学习方法**：Cellpose预训练模型
- **交互式界面**：点击查看细胞参数
- **Web界面**：实时调整参数

#### 参数说明

**传统方法参数：**
- 最小面积: 80 像素²
- 最大面积: 1000 像素²
- 最大灰度值: 120
- 最小圆度: 0.20

**Cellpose参数：**
- 预期细胞直径: 30 像素
- 流场阈值: 0.4
- 细胞概率阈值: 0.0

#### 两种方法对比

| 特性 | 传统方法 | Cellpose深度学习 |
|------|---------|-----------------|
| **速度** | 快（秒级） | 较慢（分钟级） |
| **准确性** | 中等 | 高 |
| **粘连细胞** | 有限 | 更好 |
| **参数调整** | 需要手动 | 自动学习 |
| **依赖** | 轻量 | 需要PyTorch |
| **首次运行** | 即时 | 需下载模型(1.15GB) |

#### 注意事项

1. **Cellpose首次运行**会自动下载预训练模型（约1.15GB）
2. **图像格式**支持：PNG, JPG, JPEG, TIF, TIFF
3. **图像要求**：细胞为暗色，背景为亮色
4. **性能优化**：Cellpose在多核CPU上运行更快
   ```bash
   export OMP_NUM_THREADS=8
   export MKL_NUM_THREADS=8
   ```

---

## 🖥️ 环境要求

### MATLAB 部分
- MATLAB R2018b 或更高版本
- Image Processing Toolbox

### Python 部分
- Python 3.8 或更高版本
- 至少 4GB RAM
- 至少 2GB 可用空间（Cellpose需要）

---

## 📝 参考

- Gonzalez & Woods, *Digital Image Processing*, 4th Edition
- Cellpose: https://github.com/MouseLand/cellpose
