"""
细胞分割程序
功能：
1. 对细胞图像进行分割
2. 点击任意细胞显示其参数（长轴、短轴、圆度、均值、方差）
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from skimage import measure, morphology
from skimage.segmentation import watershed
from skimage.color import label2rgb
from scipy import ndimage as ndi

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CellSegmentation:
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

    def preprocess(self):
        """图像预处理：去噪、增强对比度"""
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)

        # 直方图均衡化增强对比度
        enhanced = cv2.equalizeHist(blurred)

        return enhanced

    def segment_cells(self):
        """细胞分割：使用Otsu阈值 + 形态学处理 + 分水岭算法"""
        # 预处理
        preprocessed = self.preprocess()

        # Otsu自动阈值分割
        _, binary = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 形态学操作：去除小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # 距离变换找到前景区域
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # 找到未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 标记连通区域
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # 应用分水岭算法
        markers = cv2.watershed(self.original_image, markers)

        # 创建标记图像（去除边界标记-1）
        self.labeled_image = markers.copy()
        self.labeled_image[markers == -1] = 0
        self.labeled_image[markers == 1] = 0  # 去除背景

        # 重新标记，使标签连续
        self.labeled_image = measure.label(self.labeled_image > 1)

        return self.labeled_image

    def calculate_properties(self):
        """计算每个细胞的属性"""
        if self.labeled_image is None:
            self.segment_cells()

        # 使用skimage的regionprops计算区域属性
        self.regions = measure.regionprops(self.labeled_image, intensity_image=self.gray_image)

        return self.regions

    def get_cell_parameters(self, region):
        """获取单个细胞的参数"""
        # 长轴长度（使用新API）
        major_axis = getattr(region, 'axis_major_length', region.major_axis_length)

        # 短轴长度（使用新API）
        minor_axis = getattr(region, 'axis_minor_length', region.minor_axis_length)

        # 圆度 = 4π×面积 / 周长²，完美圆形为1
        perimeter = region.perimeter
        area = region.area
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # 细胞内像素均值（使用新API）
        mean_intensity = getattr(region, 'intensity_mean', region.mean_intensity)

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
        if x < 0 or x >= self.labeled_image.shape[1] or y < 0 or y >= self.labeled_image.shape[0]:
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

        # 使用新API获取轴长度
        minor_axis = getattr(region, 'axis_minor_length', region.minor_axis_length)
        major_axis = getattr(region, 'axis_major_length', region.major_axis_length)

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

        # 创建彩色标签图像用于显示
        label_image_colored = label2rgb(self.labeled_image, bg_label=0)

        # 创建图形
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 10))

        # 显示原图和分割结果的叠加
        self.ax.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        self.ax.imshow(label_image_colored, alpha=0.4)

        # 标记每个细胞的中心
        for region in self.regions:
            y0, x0 = region.centroid
            self.ax.plot(x0, y0, 'r+', markersize=10, markeredgewidth=2)
            self.ax.text(x0, y0-15, f'#{region.label}',
                        color='white', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7),
                        ha='center')

        self.ax.set_title('Cell Segmentation Result (Click on any cell to view parameters)', fontsize=14, fontweight='bold')
        self.ax.axis('off')

        # 连接点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        print(f"\n检测到 {len(self.regions)} 个细胞")
        print("点击任意细胞查看其参数 (Click any cell to view parameters)\n")

        plt.tight_layout()
        plt.show()

    def save_results(self, output_path='cell_segmentation_result.png'):
        """保存分割结果"""
        if self.labeled_image is None:
            self.segment_cells()

        # 创建可视化结果
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 原图
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('原始图像', fontsize=12)
        axes[0, 0].axis('off')

        # 灰度图
        axes[0, 1].imshow(self.gray_image, cmap='gray')
        axes[0, 1].set_title('灰度图像', fontsize=12)
        axes[0, 1].axis('off')

        # 分割结果
        label_image_colored = label2rgb(self.labeled_image, bg_label=0)
        axes[1, 0].imshow(label_image_colored)
        axes[1, 0].set_title(f'分割结果 ({len(self.regions)} 个细胞)', fontsize=12)
        axes[1, 0].axis('off')

        # 叠加显示
        axes[1, 1].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[1, 1].imshow(label_image_colored, alpha=0.4)
        for region in self.regions:
            y0, x0 = region.centroid
            axes[1, 1].plot(x0, y0, 'r+', markersize=8, markeredgewidth=2)
        axes[1, 1].set_title('叠加显示', fontsize=12)
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存到: {output_path}")
        plt.close()


def main():
    """主函数"""
    import sys

    # 图像路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'Picture1.png'

    print("="*60)
    print("细胞分割程序")
    print("="*60)
    print(f"正在处理图像: {image_path}\n")

    try:
        # 创建分割对象
        segmenter = CellSegmentation(image_path)

        # 执行分割
        print("正在进行细胞分割...")
        segmenter.segment_cells()

        # 计算属性
        print("正在计算细胞参数...")
        segmenter.calculate_properties()

        # 保存结果
        print("正在保存结果...")
        segmenter.save_results('cell_segmentation_result.png')

        # 显示交互式界面
        print("\n启动交互式界面...")
        segmenter.display_interactive()

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
