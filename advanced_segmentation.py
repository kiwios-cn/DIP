"""
高级图像分割模块

实现高级图像分割算法：
1. Canny边缘检测（详细实现）
2. Hough变换（直线检测、圆检测）
3. OTSU最大类间方差法（自动阈值分割）
4. 分水岭分割算法
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from scipy import ndimage


class AdvancedSegmentation:
    """高级图像分割类"""

    def __init__(self):
        self.image = None

    # ==================== Canny边缘检测（详细实现）====================

    def canny_edge_detection(self, image: np.ndarray,
                            low_threshold: int = 50,
                            high_threshold: int = 150,
                            sigma: float = 1.0) -> Tuple[np.ndarray, dict]:
        """
        Canny边缘检测（详细步骤）

        步骤：
        1. 高斯滤波去噪
        2. 计算梯度幅值和方向
        3. 非极大值抑制
        4. 双阈值检测
        5. 边缘跟踪（滞后阈值）

        Args:
            image: 输入灰度图像
            low_threshold: 低阈值
            high_threshold: 高阈值
            sigma: 高斯滤波标准差

        Returns:
            Tuple[np.ndarray, dict]:
                - 边缘图像
                - 中间结果字典（用于可视化）

        特点：
            - 最优边缘检测
            - 边缘连续
            - 单像素宽度
            - 噪声抑制好
        """
        # 步骤1：高斯滤波去噪
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        # 步骤2：计算梯度幅值和方向（使用Sobel算子）
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx) * 180 / np.pi  # 转换为度

        # 步骤3：非极大值抑制
        nms = self._non_maximum_suppression(magnitude, direction)

        # 步骤4和5：双阈值检测和边缘跟踪
        edges = self._hysteresis_thresholding(nms, low_threshold, high_threshold)

        # 保存中间结果
        intermediate = {
            'blurred': blurred,
            'gx': gx,
            'gy': gy,
            'magnitude': magnitude,
            'direction': direction,
            'nms': nms,
            'edges': edges
        }

        return edges, intermediate

    def _non_maximum_suppression(self, magnitude: np.ndarray,
                                 direction: np.ndarray) -> np.ndarray:
        """非极大值抑制"""
        h, w = magnitude.shape
        result = np.zeros((h, w), dtype=np.float32)

        # 将角度转换到0-180度
        angle = direction % 180

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # 获取梯度方向的相邻像素
                q = 255
                r = 255

                # 0度：左右方向
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # 45度：右上-左下方向
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # 90度：上下方向
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # 135度：左上-右下方向
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                # 如果当前像素是局部最大值，保留
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    result[i, j] = magnitude[i, j]

        return result

    def _hysteresis_thresholding(self, image: np.ndarray,
                                 low: float, high: float) -> np.ndarray:
        """双阈值检测和边缘跟踪"""
        h, w = image.shape
        result = np.zeros((h, w), dtype=np.uint8)

        # 强边缘
        strong = (image >= high)
        # 弱边缘
        weak = (image >= low) & (image < high)

        result[strong] = 255

        # 边缘跟踪：连接弱边缘到强边缘
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if weak[i, j]:
                    # 检查8邻域是否有强边缘
                    if np.any(strong[i-1:i+2, j-1:j+2]):
                        result[i, j] = 255
                        strong[i, j] = True  # 标记为强边缘，用于继续连接

        return result

    # ==================== Hough变换 ====================

    def hough_lines(self, edges: np.ndarray,
                   rho: float = 1,
                   theta: float = np.pi/180,
                   threshold: int = 100,
                   min_line_length: int = 50,
                   max_line_gap: int = 10) -> List[np.ndarray]:
        """
        Hough直线检测

        使用Hough变换在边缘图像中检测直线。

        Args:
            edges: 边缘图像（二值）
            rho: 距离分辨率（像素）
            theta: 角度分辨率（弧度）
            threshold: 累加器阈值
            min_line_length: 最小线段长度
            max_line_gap: 最大线段间隙

        Returns:
            List[np.ndarray]: 检测到的线段列表，每条线[x1,y1,x2,y2]

        原理：
            直线的极坐标表示：ρ = x·cos(θ) + y·sin(θ)
            在参数空间(ρ,θ)中进行投票
        """
        lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                               minLineLength=min_line_length,
                               maxLineGap=max_line_gap)

        if lines is None:
            return []

        return [line[0] for line in lines]

    def hough_circles(self, image: np.ndarray,
                     min_radius: int = 10,
                     max_radius: int = 100,
                     param1: int = 100,
                     param2: int = 30,
                     min_dist: int = 50) -> List[np.ndarray]:
        """
        Hough圆检测

        使用Hough变换检测图像中的圆。

        Args:
            image: 输入灰度图像
            min_radius: 最小圆半径
            max_radius: 最大圆半径
            param1: Canny边缘检测的高阈值
            param2: 累加器阈值（越小检测到越多圆）
            min_dist: 检测到的圆心之间的最小距离

        Returns:
            List[np.ndarray]: 检测到的圆列表，每个圆[x,y,r]

        原理：
            圆的方程：(x-a)² + (y-b)² = r²
            在参数空间(a,b,r)中进行投票
        """
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, min_dist,
                                  param1=param1, param2=param2,
                                  minRadius=min_radius, maxRadius=max_radius)

        if circles is None:
            return []

        circles = np.uint16(np.around(circles))
        return [circle for circle in circles[0, :]]

    # ==================== OTSU最大类间方差法 ====================

    def otsu_threshold(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        OTSU最大类间方差法（自动阈值分割）

        自动计算最优阈值，使类间方差最大。

        Args:
            image: 输入灰度图像

        Returns:
            Tuple[int, np.ndarray]:
                - 最优阈值
                - 二值化结果

        原理：
            1. 计算直方图
            2. 遍历所有可能的阈值
            3. 计算类间方差
            4. 选择使类间方差最大的阈值

        公式：
            σ²_between = w0 × w1 × (μ0 - μ1)²
            其中：
            w0, w1: 两类的像素比例
            μ0, μ1: 两类的平均灰度
        """
        # 计算直方图
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        # 总像素数
        total = image.size

        # 当前最优值
        best_threshold = 0
        max_variance = 0

        # 遍历所有可能的阈值
        sum_total = np.dot(np.arange(256), hist)
        sum_background = 0
        weight_background = 0

        for t in range(256):
            weight_background += hist[t]
            if weight_background == 0:
                continue

            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break

            sum_background += t * hist[t]

            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground

            # 类间方差
            variance = weight_background * weight_foreground * \
                      (mean_background - mean_foreground) ** 2

            if variance > max_variance:
                max_variance = variance
                best_threshold = t

        # 应用阈值
        _, binary = cv2.threshold(image, best_threshold, 255, cv2.THRESH_BINARY)

        return best_threshold, binary

    # ==================== 分水岭分割 ====================

    def watershed_segmentation(self, image: np.ndarray,
                              markers: Optional[np.ndarray] = None,
                              auto_markers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        分水岭分割算法

        将图像视为地形图，从标记点开始"注水"，
        在不同水域交汇处形成分水岭（边界）。

        Args:
            image: 输入彩色图像
            markers: 标记图像（可选）
            auto_markers: 是否自动生成标记

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - 分割结果
                - 标记图像

        原理：
            1. 将图像视为地形图（梯度作为高度）
            2. 从标记点开始向上"注水"
            3. 不同区域的水汇合处形成分水岭

        步骤：
            1. 去噪
            2. 提取前景标记
            3. 提取背景标记
            4. 应用分水岭算法
        """
        if len(image.shape) == 2:
            # 灰度图转彩色
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if markers is None and auto_markers:
            # 自动生成标记
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 1. OTSU二值化
            _, binary = self.otsu_threshold(gray)

            # 2. 形态学操作去噪
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

            # 3. 确定背景区域
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # 4. 距离变换找前景
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

            # 5. 找到不确定区域
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # 6. 标记连通区域
            _, markers = cv2.connectedComponents(sure_fg)

            # 7. 标记背景为1
            markers = markers + 1

            # 8. 不确定区域标记为0
            markers[unknown == 255] = 0

        # 应用分水岭算法
        markers_copy = markers.copy()
        cv2.watershed(image, markers_copy)

        # 提取边界（标记为-1）
        result = image.copy()
        result[markers_copy == -1] = [0, 0, 255]  # 红色边界

        return result, markers

    # ==================== 辅助方法 ====================

    def load_image(self, path: str, grayscale: bool = True) -> bool:
        """加载图像"""
        if grayscale:
            self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            self.image = cv2.imread(path)
        return self.image is not None

    def visualize_canny_steps(self, image: np.ndarray,
                             save_path: Optional[str] = None):
        """可视化Canny边缘检测的各个步骤"""
        edges, intermediate = self.canny_edge_detection(image, 50, 150)

        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()

        # 原图
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('1. 原始图像', fontsize=11)
        axes[0].axis('off')

        # 高斯滤波
        axes[1].imshow(intermediate['blurred'], cmap='gray')
        axes[1].set_title('2. 高斯滤波', fontsize=11)
        axes[1].axis('off')

        # X方向梯度
        axes[2].imshow(intermediate['gx'], cmap='gray')
        axes[2].set_title('3. X方向梯度 (Gx)', fontsize=11)
        axes[2].axis('off')

        # Y方向梯度
        axes[3].imshow(intermediate['gy'], cmap='gray')
        axes[3].set_title('4. Y方向梯度 (Gy)', fontsize=11)
        axes[3].axis('off')

        # 梯度幅值
        axes[4].imshow(intermediate['magnitude'], cmap='gray')
        axes[4].set_title('5. 梯度幅值', fontsize=11)
        axes[4].axis('off')

        # 梯度方向
        axes[5].imshow(intermediate['direction'], cmap='hsv')
        axes[5].set_title('6. 梯度方向', fontsize=11)
        axes[5].axis('off')

        # 非极大值抑制
        axes[6].imshow(intermediate['nms'], cmap='gray')
        axes[6].set_title('7. 非极大值抑制', fontsize=11)
        axes[6].axis('off')

        # 最终边缘
        axes[7].imshow(edges, cmap='gray')
        axes[7].set_title('8. 双阈值+边缘跟踪', fontsize=11)
        axes[7].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Canny步骤可视化已保存到: {save_path}")

        plt.show()


def main():
    """演示程序"""
    print("高级图像分割演示程序")
    print("=" * 50)

    seg = AdvancedSegmentation()

    # 创建测试图像
    test_image = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 50), (200, 200), 200, 3)
    cv2.circle(test_image, (180, 100), 30, 200, 2)
    cv2.line(test_image, (30, 220), (220, 220), 200, 2)

    cv2.imwrite('advanced_seg_test.png', test_image)

    # 测试1: Canny边缘检测步骤可视化
    print("\n测试1: Canny边缘检测步骤可视化")
    seg.visualize_canny_steps(test_image, save_path='canny_steps.png')

    # 测试2: Hough直线检测
    print("\n测试2: Hough直线检测")
    edges, _ = seg.canny_edge_detection(test_image)
    lines = seg.hough_lines(edges, threshold=50, min_line_length=50)

    line_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite('hough_lines.png', line_image)
    print(f"检测到 {len(lines)} 条直线")

    # 测试3: OTSU阈值分割
    print("\n测试3: OTSU最大类间方差法")
    threshold, binary = seg.otsu_threshold(test_image)
    print(f"OTSU最优阈值: {threshold}")
    cv2.imwrite('otsu_result.png', binary)

    print("\n✅ 演示完成！")


if __name__ == '__main__':
    main()
