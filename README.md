# 细胞分割系统

基于图像处理和深度学习的细胞分割工具，提供命令行和Web界面两种使用方式。

## 📋 功能特点

- **传统方法**：Otsu阈值 + 分水岭算法
- **深度学习方法**：Cellpose预训练模型
- **交互式界面**：点击查看细胞参数
- **Web界面**：实时调整参数
- **批量处理**：支持多图像处理

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行Web版本

**传统方法版本：**
```bash
streamlit run app.py
```

**Cellpose深度学习版本：**
```bash
streamlit run app_cellpose.py
```

### 3. 运行命令行版本

**传统方法版本：**
```bash
python cell_segmentation_improved.py Picture1.png
```

**Cellpose深度学习版本：**
```bash
python cell_segmentation_cellpose.py Picture1.png
```

## 📁 文件说明

### 主要程序文件

| 文件名 | 说明 | 类型 |
|--------|------|------|
| `app.py` | 传统方法Web界面 | Web应用 |
| `app_cellpose.py` | Cellpose深度学习Web界面 | Web应用 |
| `cell_segmentation_improved.py` | 传统方法命令行版本 | 命令行工具 |
| `cell_segmentation_cellpose.py` | Cellpose命令行版本 | 命令行工具 |

### 配置文件

| 文件名 | 说明 |
|--------|------|
| `requirements.txt` | Python依赖包列表 |
| `README.md` | 项目说明文档 |

### 示例文件

| 文件名 | 说明 |
|--------|------|
| `Picture1.png` | 示例细胞图像 |
| `generate_result.py` | 批量生成结果脚本 |

## 🔧 参数说明

### 传统方法参数

- **最小面积**: 80 像素² (过滤小噪点)
- **最大面积**: 1000 像素² (过滤大背景区域)
- **最大灰度值**: 120 (过滤高灰度背景)
- **最小圆度**: 0.20 (过滤不规则形状)

### Cellpose参数

- **预期细胞直径**: 30 像素 (影响分割尺度)
- **流场阈值**: 0.4 (控制分割严格程度)
- **细胞概率阈值**: 0.0 (判断是否为细胞)

## 📊 两种方法对比

| 特性 | 传统方法 | Cellpose深度学习 |
|------|---------|-----------------|
| **速度** | 快（秒级） | 较慢（分钟级） |
| **准确性** | 中等 | 高 |
| **粘连细胞** | 有限 | 更好 |
| **参数调整** | 需要手动 | 自动学习 |
| **依赖** | 轻量 | 需要PyTorch |
| **首次运行** | 即时 | 需下载模型(1.15GB) |

## 💡 使用建议

### 选择传统方法，如果：
- 需要快速处理
- 细胞形态规则
- 计算资源有限
- 不想下载大模型

### 选择Cellpose，如果：
- 需要高精度
- 细胞有粘连
- 细胞形态复杂
- 有足够的计算资源

## 🖥️ 系统要求

- **Python**: 3.8 或更高版本
- **内存**: 至少 4GB RAM
- **存储**: 至少 2GB 可用空间（Cellpose需要）
- **操作系统**: Windows / macOS / Linux

## 📖 使用示例

### Web界面使用

1. 启动Web应用
2. 上传细胞图像
3. 调整参数（可选）
4. 点击"开始分割"
5. 查看结果和统计信息
6. 选择细胞查看详细参数

### 命令行使用

```bash
# 使用传统方法
python cell_segmentation_improved.py your_image.png

# 使用Cellpose
python cell_segmentation_cellpose.py your_image.png
```

## ⚠️ 注意事项

1. **Cellpose首次运行**会自动下载预训练模型（约1.15GB），需要网络连接
2. **图像格式**支持：PNG, JPG, JPEG, TIF, TIFF
3. **图像要求**：细胞为暗色，背景为亮色
4. **性能优化**：Cellpose在多核CPU上运行更快，建议设置环境变量：
   ```bash
   export OMP_NUM_THREADS=8
   export MKL_NUM_THREADS=8
   ```

## 🐛 常见问题

### Q: Web界面无法启动？
A: 确保已安装streamlit：`pip install streamlit`

### Q: Cellpose运行很慢？
A: 首次运行需要下载模型。后续运行会快很多。可以设置多线程优化（见注意事项）。

### Q: 检测不到细胞？
A: 尝试调整参数，特别是灰度阈值和面积范围。

### Q: 检测到太多背景区域？
A: 降低"最大灰度值"参数，过滤高灰度背景。

## 📦 分享给他人

### 方式1：完整项目包（推荐）

打包以下文件发送：
```
项目文件夹/
├── app.py                              # 传统方法Web版
├── app_cellpose.py                     # Cellpose Web版
├── cell_segmentation_improved.py       # 传统方法命令行版
├── cell_segmentation_cellpose.py       # Cellpose命令行版
├── requirements.txt                    # 依赖列表
├── README.md                           # 使用说明
└── Picture1.png                        # 示例图像（可选）
```

接收者需要：
1. 安装Python 3.8+
2. 运行 `pip install -r requirements.txt`
3. 运行 `streamlit run app.py` 或其他版本

### 方式2：Docker容器（高级）

创建Docker镜像，一键部署，无需配置环境。

### 方式3：云端部署

部署到Streamlit Cloud，分享网址即可使用，无需安装。

## 📝 更新日志

### v1.0.0 (2024-05)
- 实现传统方法（Otsu + 分水岭）
- 实现Cellpose深度学习方法
- 提供命令行和Web两种界面
- 支持交互式参数调整
- 优化CPU多线程性能

## 📄 许可证

本项目仅供学习和研究使用。

## 👥 贡献

欢迎提交问题和改进建议！

## 📧 联系方式

如有问题，请通过GitHub Issues联系。
