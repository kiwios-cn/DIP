"""
细胞分割程序 - 优化版本
改进：
1. 更好的预处理（自适应滤波）
2. 改进的阈值方法（多阈值+形态学重建）
3. 优化的分水岭算法参数
4. 更准确的细胞检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from skimage import measure, morphology, filters
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CellSegmentationImproved:
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
        """改进的预处理：双边滤波保留边缘 + CLAHE增强对比度"""
        # 双边滤波：去噪同时保留边缘
        bilateral = cv2.bilateralFilter(self.gray_image, 9, 75, 75)

        # CLAHE (对比度受限的自适应直方图均衡化)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(bilateral)

        return enhanced

    def segment_cells(self):
        """改进的细胞分割算法"""
        # 预处理
        preprocessed = self.preprocess()

        # 使用多种方法结合进行阈值分割
        # 方法1: Otsu阈值
        _, binary_otsu = cv2.threshold(preprocessed, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 方法2: 自适应阈值（处理光照不均）
        binary_adaptive = cv2.adaptiveThreshold(preprocessed, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 21, 5)

        # 结合两种方法
        binary = cv2.bitwise_and(binary_otsu, binary_adaptive)

        # 形态学操作：去除小噪点（减少开运算迭代以保留更多边缘）
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # 填充小孔洞
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=2)

        # 去除边界上的对象
        binary = self.remove_border_objects(binary)

        # 去除过小的对象（噪声）
        binary = self.remove_small_objects(binary, min_size=100)

        # 距离变换
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # 使用局部最大值找到细胞中心（第一阶段：适度提高参数减少误分割）
        local_max = peak_local_max(dist_transform, min_distance=15,
                                   threshold_rel=0.15, labels=binary)

        # 创建标记
        markers = np.zeros_like(binary, dtype=np.int32)
        for idx, (y, x) in enumerate(local_max, start=1):
            markers[y, x] = idx

        # 扩展标记（减小扩展半径以避免过度合并）
        markers = morphology.dilation(markers, morphology.disk(2))

        # 应用分水岭算法
        # 使用梯度图像作为分水岭的输入（更好的边界）
        gradient = cv2.morphologyEx(preprocessed, cv2.MORPH_GRADIENT,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # 分水岭
        markers = watershed(gradient, markers, mask=binary)

        # 标记图像
        self.labeled_image = markers

        return self.labeled_image

    def remove_border_objects(self, binary):
        """去除接触边界的对象"""
        # 标记连通区域
        labeled = measure.label(binary)

        # 找到接触边界的标签
        border_labels = set()
        h, w = binary.shape

        # 检查四条边
        border_labels.update(labeled[0, :])  # 上边
        border_labels.update(labeled[-1, :])  # 下边
        border_labels.update(labeled[:, 0])  # 左边
        border_labels.update(labeled[:, -1])  # 右边

        # 移除背景标签0
        border_labels.discard(0)

        # 创建掩码
        mask = np.ones_like(binary)
        for label in border_labels:
            mask[labeled == label] = 0

        return binary * mask

    def remove_small_objects(self, binary, min_size=100):
        """去除小于指定大小的对象"""
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        mask = np.zeros_like(binary)
        for region in regions:
            if region.area >= min_size:
                coords = region.coords
                for coord in coords:
                    mask[coord[0], coord[1]] = 1

        return binary * mask

    def merge_small_fragments(self, labels_image):
        """第二阶段：合并小碎片到相邻区域"""
        merged_labels = labels_image.copy()
        regions = measure.regionprops(labels_image, intensity_image=self.gray_image)

        # 找出所有小碎片（面积<100）
        small_fragments = []
        region_dict = {}  # label -> region

        for region in regions:
            region_dict[region.label] = region
            if region.area < 100:
                small_fragments.append(region)

        print(f"发现 {len(small_fragments)} 个小碎片待合并")

        merged_count = 0
        for fragment in small_fragments:
            fragment_label = fragment.label
            fragment_intensity = getattr(fragment, 'intensity_mean',
                                        getattr(fragment, 'mean_intensity', 0))

            # 找到相邻区域
            # 膨胀碎片区域1个像素，看接触到哪些其他区域
            fragment_mask = (labels_image == fragment_label)
            dilated = morphology.binary_dilation(fragment_mask, morphology.disk(1))

            # 找到膨胀区域内的其他标签
            neighbor_labels = np.unique(labels_image[dilated])
            neighbor_labels = neighbor_labels[neighbor_labels != fragment_label]
            neighbor_labels = neighbor_labels[neighbor_labels != 0]  # 排除背景

            if len(neighbor_labels) == 0:
                continue

            # 找到最佳合并候选（灰度最接近的邻居）
            best_neighbor = None
            min_intensity_diff = float('inf')

            for neighbor_label in neighbor_labels:
                if neighbor_label not in region_dict:
                    continue

                neighbor_region = region_dict[neighbor_label]
                neighbor_intensity = getattr(neighbor_region, 'intensity_mean',
                                            getattr(neighbor_region, 'mean_intensity', 0))

                intensity_diff = abs(fragment_intensity - neighbor_intensity)
                combined_area = fragment.area + neighbor_region.area

                # 检查合并条件：灰度差<20 且合并后总面积<1000
                if intensity_diff < 20 and combined_area < 1000:
                    if intensity_diff < min_intensity_diff:
                        min_intensity_diff = intensity_diff
                        best_neighbor = neighbor_label

            # 执行合并
            if best_neighbor is not None:
                merged_labels[fragment_mask] = best_neighbor
                merged_count += 1

        print(f"成功合并 {merged_count} 个小碎片")
        return merged_labels

    def calculate_properties(self):
        """计算每个细胞的属性"""
        if self.labeled_image is None:
            self.segment_cells()

        # 第二阶段：合并小碎片
        self.labeled_image = self.merge_small_fragments(self.labeled_image)

        # 使用skimage的regionprops计算区域属性
        self.regions = measure.regionprops(self.labeled_image,
                                          intensity_image=self.gray_image)

        print(f"合并后检测到 {len(self.regions)} 个区域")

        # 过滤掉不符合标准的区域
        filtered_regions = []
        for region in self.regions:
            area = region.area
            perimeter = region.perimeter

            # 计算圆度
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0

            # 计算平均灰度（兼容不同版本的API）
            mean_intensity = getattr(region, 'intensity_mean',
                                   getattr(region, 'mean_intensity', 0))

            # 过滤标准（混合策略 + 背景过滤）：
            # - 面积: 80 ~ 1000 像素²
            # - 平均灰度: <= 120 (过滤高灰度背景区域)
            # - 圆度: >= 0.20
            if (80 <= area <= 1000 and
                mean_intensity <= 120 and
                circularity >= 0.20):
                filtered_regions.append(region)

        print(f"过滤后保留 {len(filtered_regions)} 个细胞")
        print(f"过滤标准: 面积[80-1000], 灰度≤120, 圆度≥0.20")

        self.regions = filtered_regions

        return self.regions

    def get_cell_parameters(self, region):
        """获取单个细胞的参数"""
        # 长轴长度（使用新API）
        major_axis = getattr(region, 'axis_major_length',
                           getattr(region, 'major_axis_length', 0))

        # 短轴长度（使用新API）
        minor_axis = getattr(region, 'axis_minor_length',
                           getattr(region, 'minor_axis_length', 0))

        # 圆度 = 4π×面积 / 周长²
        perimeter = region.perimeter
        area = region.area
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # 细胞内像素均值（使用新API）
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

        # 使用新API获取轴长度
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

        # 创建彩色标签图像用于显示
        label_image_colored = label2rgb(self.labeled_image, bg_label=0,
                                       image=self.gray_image, alpha=0.3)

        # 创建图形
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 12))

        # 显示原图和分割结果的叠加
        self.ax.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        self.ax.imshow(label_image_colored, alpha=0.5)

        # 标记每个细胞的轮廓
        for region in self.regions:
            # 绘制轮廓
            contours = measure.find_contours(self.labeled_image == region.label, 0.5)
            for contour in contours:
                self.ax.plot(contour[:, 1], contour[:, 0], 'g-', linewidth=0.8, alpha=0.8)

        self.ax.set_title('Cell Segmentation Result (Click on any cell to view parameters)',
                         fontsize=14, fontweight='bold')
        self.ax.axis('off')

        # 连接点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        print(f"\n检测到 {len(self.regions)} 个细胞")
        print("点击任意细胞查看其参数 (Click any cell to view parameters)\n")

        plt.tight_layout()
        plt.show()

    def save_results(self, output_path='cell_segmentation_improved_result.png'):
        """保存分割结果"""
        if self.labeled_image is None:
            self.segment_cells()

        # 创建可视化结果
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 原图
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # 预处理后的图像
        preprocessed = self.preprocess()
        axes[0, 1].imshow(preprocessed, cmap='gray')
        axes[0, 1].set_title('Preprocessed (CLAHE)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # 二值图
        _, binary = cv2.threshold(preprocessed, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        axes[0, 2].imshow(binary, cmap='gray')
        axes[0, 2].set_title('Binary Image', fontsize=12, fontweight='bold')
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

    # 图像路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'Picture1.png'

    print("="*60)
    print("细胞分割程序 - 优化版本")
    print("="*60)
    print(f"正在处理图像: {image_path}\n")

    try:
        # 创建分割对象
        segmenter = CellSegmentationImproved(image_path)

        # 执行分割
        print("正在进行细胞分割...")
        segmenter.segment_cells()

        # 计算属性
        print("正在计算细胞参数...")
        segmenter.calculate_properties()

        # 保存结果
        print("正在保存结果...")
        segmenter.save_results('cell_segmentation_improved_result.png')

        # 显示交互式界面
        print("\n启动交互式界面...")
        segmenter.display_interactive()

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
