"""
细胞分割程序 - Cellpose深度学习版本
使用Cellpose预训练模型进行细胞分割
优势：
1. 更好地处理粘连细胞
2. 更准确的边界检测
3. 无需手动调整参数
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CellSegmentationCellpose:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.labeled_image = None
        self.regions = None
        self.fig = None
        self.ax = None
        self.model = None

    def load_cellpose_model(self, model_type='cyto', gpu=False):
        """加载Cellpose模型"""
        try:
            from cellpose import models
            print(f"正在加载Cellpose模型: {model_type}")
            self.model = models.CellposeModel(gpu=gpu, model_type=model_type)
            print("模型加载成功")
            return True
        except ImportError:
            print("错误: 未安装cellpose库")
            print("请运行: pip install cellpose")
            return False
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

    def segment_cells(self, diameter=30, flow_threshold=0.4, cellprob_threshold=0.0):
        """
        使用Cellpose进行细胞分割

        参数:
        - diameter: 预期细胞直径（像素），None表示自动估计
        - flow_threshold: 流场阈值，越高越严格（默认0.4）
        - cellprob_threshold: 细胞概率阈值（默认0.0）
        """
        if self.model is None:
            if not self.load_cellpose_model():
                raise RuntimeError("无法加载Cellpose模型")

        print(f"正在使用Cellpose分割细胞...")
        print(f"参数: diameter={diameter}, flow_threshold={flow_threshold}, cellprob_threshold={cellprob_threshold}")

        # Cellpose需要RGB图像或灰度图像
        # 使用灰度图像进行分割
        result = self.model.eval(
            self.gray_image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=[0, 0]  # [0,0]表示灰度图像
        )

        # 兼容不同版本的返回值
        if len(result) == 4:
            masks, flows, styles, diams = result
        else:
            masks, flows, styles = result
            diams = diameter  # 使用输入的直径

        self.labeled_image = masks
        print(f"Cellpose检测到 {masks.max()} 个区域")
        if isinstance(diams, (int, float)):
            print(f"估计的细胞直径: {diams:.1f} 像素")
        else:
            print(f"使用的细胞直径: {diameter} 像素")

        return self.labeled_image

    def calculate_properties(self, min_area=80, max_area=1000,
                           max_intensity=120, min_circularity=0.20):
        """
        计算细胞属性并过滤

        参数:
        - min_area: 最小面积
        - max_area: 最大面积
        - max_intensity: 最大灰度值（过滤背景）
        - min_circularity: 最小圆度
        """
        if self.labeled_image is None:
            self.segment_cells()

        from skimage import measure

        self.regions = measure.regionprops(self.labeled_image,
                                          intensity_image=self.gray_image)

        print(f"Cellpose原始检测: {len(self.regions)} 个区域")

        # 过滤不符合标准的细胞
        filtered_regions = []
        for region in self.regions:
            area = region.area
            perimeter = region.perimeter

            # 计算圆度
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0

            # 计算平均灰度
            mean_intensity = getattr(region, 'intensity_mean',
                                   getattr(region, 'mean_intensity', 0))

            # 过滤标准
            if (min_area <= area <= max_area and
                mean_intensity <= max_intensity and
                circularity >= min_circularity):
                filtered_regions.append(region)

        print(f"过滤后保留 {len(filtered_regions)} 个细胞")
        print(f"过滤标准: 面积[{min_area}-{max_area}], 灰度≤{max_intensity}, 圆度≥{min_circularity}")

        self.regions = filtered_regions

        return self.regions

    def get_cell_parameters(self, region):
        """获取单个细胞的参数"""
        # 长轴长度
        major_axis = getattr(region, 'axis_major_length',
                           getattr(region, 'major_axis_length', 0))

        # 短轴长度
        minor_axis = getattr(region, 'axis_minor_length',
                           getattr(region, 'minor_axis_length', 0))

        # 圆度
        perimeter = region.perimeter
        area = region.area
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # 细胞内像素均值
        mean_intensity = getattr(region, 'intensity_mean',
                               getattr(region, 'mean_intensity', 0))

        # 细胞内像素方差
        coords = region.coords
        pixel_values = [self.gray_image[coord[0], coord[1]] for coord in coords]
        variance = np.var(pixel_values)

        return {
            '长轴长度': major_axis,
            '短轴长度': minor_axis,
            '圆度': circularity,
            '均值': mean_intensity,
            '方差': variance,
            '面积': area,
            '周长': perimeter
        }

    def onclick(self, event):
        """鼠标点击事件处理"""
        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        # 检查点击位置是否在图像范围内
        if (x < 0 or x >= self.labeled_image.shape[1] or
            y < 0 or y >= self.labeled_image.shape[0]):
            return

        # 获取点击位置的细胞标签
        label = self.labeled_image[y, x]

        if label == 0:
            print("点击位置不在细胞区域")
            return

        # 找到对应的region
        region = None
        for r in self.regions:
            if r.label == label:
                region = r
                break

        if region is None:
            return

        # 获取细胞参数
        params = self.get_cell_parameters(region)

        # 清除之前的标注
        for artist in self.ax.patches[:]:
            artist.remove()
        for artist in self.ax.texts[:]:
            artist.remove()

        # 绘制椭圆标注
        y0, x0 = region.centroid
        orientation = region.orientation

        minor_axis = getattr(region, 'axis_minor_length',
                           getattr(region, 'minor_axis_length', 0))
        major_axis = getattr(region, 'axis_major_length',
                           getattr(region, 'major_axis_length', 0))

        ellipse = Ellipse(
            xy=(x0, y0),
            width=minor_axis,
            height=major_axis,
            angle=np.degrees(orientation),
            edgecolor='red',
            facecolor='none',
            linewidth=2
        )
        self.ax.add_patch(ellipse)

        # 显示参数信息
        info_text = (
            f"Cell #{label}\n"
            f"Major: {params['长轴长度']:.2f} px\n"
            f"Minor: {params['短轴长度']:.2f} px\n"
            f"Circularity: {params['圆度']:.3f}\n"
            f"Area: {params['面积']:.0f} px²\n"
            f"Perimeter: {params['周长']:.2f} px\n"
            f"Mean: {params['均值']:.2f}\n"
            f"Variance: {params['方差']:.2f}"
        )

        self.ax.text(
            x0 + major_axis/2 + 10, y0,
            info_text,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            fontsize=9,
            verticalalignment='center'
        )

        # 在控制台也打印信息
        print(f"\n{'='*50}")
        print(f"细胞 #{label} 的参数：")
        print(f"{'='*50}")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"{key:10s}: {value:10.2f}")
            else:
                print(f"{key:10s}: {value}")
        print(f"{'='*50}\n")

        self.fig.canvas.draw()

    def display_interactive(self):
        """显示交互式分割结果"""
        if self.regions is None:
            self.calculate_properties()

        from skimage.color import label2rgb

        # 创建彩色标签图像
        label_image_colored = label2rgb(self.labeled_image, bg_label=0,
                                       image=self.gray_image, alpha=0.3)

        # 创建图形
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 12))

        # 显示原图和分割结果的叠加
        self.ax.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        self.ax.imshow(label_image_colored, alpha=0.5)

        # 标记每个细胞的轮廓
        from skimage import measure
        for region in self.regions:
            contours = measure.find_contours(self.labeled_image == region.label, 0.5)
            for contour in contours:
                self.ax.plot(contour[:, 1], contour[:, 0], 'g-', linewidth=0.8, alpha=0.8)

        self.ax.set_title('Cellpose Segmentation Result (Click on any cell to view parameters)',
                         fontsize=14, fontweight='bold')
        self.ax.axis('off')

        # 连接点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        print(f"\n检测到 {len(self.regions)} 个细胞")
        print("点击任意细胞查看其参数 (Click any cell to view parameters)\n")

        plt.tight_layout()
        plt.show()

    def save_results(self, output_path='cell_segmentation_cellpose_result.png'):
        """保存分割结果"""
        if self.labeled_image is None:
            self.segment_cells()

        from skimage.color import label2rgb
        from skimage import measure

        # 创建可视化结果
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 原图
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # 灰度图
        axes[0, 1].imshow(self.gray_image, cmap='gray')
        axes[0, 1].set_title('Grayscale', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Cellpose masks
        axes[0, 2].imshow(self.labeled_image, cmap='nipy_spectral')
        axes[0, 2].set_title(f'Cellpose Masks ({self.labeled_image.max()} regions)',
                            fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # 分割结果（彩色标签）
        label_image_colored = label2rgb(self.labeled_image, bg_label=0)
        axes[1, 0].imshow(label_image_colored)
        axes[1, 0].set_title(f'Segmentation ({len(self.regions)} cells)',
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        # 叠加显示
        axes[1, 1].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[1, 1].imshow(label_image_colored, alpha=0.4)
        for region in self.regions:
            y0, x0 = region.centroid
            axes[1, 1].plot(x0, y0, 'r+', markersize=10, markeredgewidth=2)
        axes[1, 1].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        # 轮廓显示
        contour_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB).copy()
        for region in self.regions:
            contours = measure.find_contours(self.labeled_image == region.label, 0.5)
            for contour in contours:
                contour = contour.astype(np.int32)
                for i in range(len(contour)-1):
                    cv2.line(contour_image,
                            (contour[i, 1], contour[i, 0]),
                            (contour[i+1, 1], contour[i+1, 0]),
                            (0, 255, 0), 1)
        axes[1, 2].imshow(contour_image)
        axes[1, 2].set_title('Contours', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存到: {output_path}")
        plt.close()


def main():
    """主函数"""
    import sys
    import os

    # 优化CPU多线程性能
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'

    try:
        import torch
        torch.set_num_threads(8)
        print(f"PyTorch线程数设置为: {torch.get_num_threads()}")
    except:
        pass

    # 图像路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'Picture1.png'

    print("="*60)
    print("细胞分割程序 - Cellpose深度学习版本")
    print("="*60)
    print(f"正在处理图像: {image_path}\n")

    try:
        # 创建分割对象
        segmenter = CellSegmentationCellpose(image_path)

        # 加载模型
        if not segmenter.load_cellpose_model(model_type='cyto', gpu=False):
            print("\n无法加载Cellpose模型，请确保已安装cellpose库")
            print("安装命令: pip install cellpose")
            return

        # 执行分割
        print("\n正在进行细胞分割...")
        segmenter.segment_cells(diameter=30, flow_threshold=0.4, cellprob_threshold=0.0)

        # 计算属性
        print("\n正在计算细胞参数...")
        segmenter.calculate_properties()

        # 保存结果
        print("\n正在保存结果...")
        segmenter.save_results('cell_segmentation_cellpose_result.png')

        # 显示交互式界面
        print("\n启动交互式界面...")
        segmenter.display_interactive()

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
