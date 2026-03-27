# DIP - Digital Image Processing

MATLAB 数字图像处理实验代码，涵盖空间域与频率域的经典方法。

## 文件说明

| 文件 | 内容 |
|------|------|
| `my_median.m` | 手动实现中值滤波（不使用 `medfilt2`） |
| `my_histeq.m` | 手动实现直方图均衡化（不使用 `histeq`） |
| `my_laplacian.m` | 拉普拉斯锐化增强 |
| `fft_demo.m` | FFT 频域分析演示（旋转/平移/尺度/直流/低中高频编辑） |

---

## 空间域处理

### 1. 中值滤波 `my_median.m`

去除椒盐噪声，保留边缘。

**原理：** 对每个像素取其邻域窗口内像素值的中位数代替原值。

```matlab
out = my_median(img);          % 默认 3×3 窗口
out = my_median(img, 5);       % 5×5 窗口（去噪更强）
```

**关键步骤：**
1. `padarray(..., 'replicate')` — 边缘复制填充，避免黑边
2. 双重循环滑动窗口，`median(patch(:))` 取中位数
3. 首尾 `double ↔ uint8` 转换防止溢出

---

### 2. 直方图均衡化 `my_histeq.m`

增强对比度，使暗图变亮、细节更清晰。

**原理：** 将像素值的分布从集中拉伸为均匀分布。

```matlab
out = my_histeq(img);
```

**核心公式：**

$$s = \left\lfloor \frac{CDF(r) - CDF_{min}}{N - CDF_{min}} \times 255 + 0.5 \right\rfloor$$

**关键步骤：**
1. 统计直方图 `h[256]`
2. 计算累积分布函数 CDF
3. 用 CDF 构建映射查找表
4. 按查找表替换所有像素

> ⚠️ 彩色图不能直接均衡化（会色偏），需先转 HSV，只对 V 通道操作。

---

### 3. 拉普拉斯锐化 `my_laplacian.m`

增强边缘，使图像更清晰。

**原理：** 用拉普拉斯算子提取边缘/细节，叠加回原图。

```matlab
out = my_laplacian(img);       % 默认 k=1.0
out = my_laplacian(img, 1.5);  % 增强更强
```

**卷积核：**
```
 0  -1   0
-1   4  -1
 0  -1   0
```

**关键代码：**
```matlab
out = uint8(min(max(img_d - k * lap, 0), 255));
%                ↑ 截上限255  ↑ 截下限0   ↑ 锐化
```

---

## 频率域处理

### 4. FFT 频域分析 `fft_demo.m`

演示傅里叶变换的五大性质与频率分量编辑。

```matlab
im = im2double(rgb2gray(imread('your_image.jpg')));
demo_rotation(im, 45);     % 旋转性质
demo_translation(im);      % 平移性质
demo_scale(im);            % 尺度变换
demo_dc(im);               % 直流分量
demo_freq_edit(im);        % 低/中/高频编辑
```

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

**⚠️ 使用 FFT 的注意事项：**
- `fftshift` 和 `ifftshift` 必须配对
- 逆变换前必须先 `ifftshift`，再 `ifft2`
- 逆变换后用 `real(...)` 消除浮点误差虚部
- 频率阈值用相对比例（`min(H,W) * 0.08`），对不同分辨率自适应

---

## 环境要求

- MATLAB R2018b 或更高版本
- Image Processing Toolbox（`padarray`、`imresize` 等函数）

## 参考

- Gonzalez & Woods, *Digital Image Processing*, 4th Edition
