"""
边缘检测模块

实现图像分割相关的检测算法：
1. 点检测 (Point Detection)
2. 线检测 (Line Detection)
3. 边缘检测：
   - Roberts算子
   - Prewitt算子
   - Sobel算子
   - Canny边缘检测
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Literal
import matplotlib.pyplot as plt


class EdgeDetection:
    """边缘检测类"""

    def __init__(self):
        self.image = None

    # ==================== 点检测 ====================

    def point_detection(self, image: np.ndarray,
                       threshold: float = 0.5) -> np.ndarray:
        """
        点检测

        检测图像中的孤立点（与周围像素差异大的点）。

        使用拉普拉斯算子的变体：
        [-1 -1 -1]
        [-1  8 -1]
        [-1 -1 -1]

        Args:
            image: 输入灰度图像
            threshold: 阈值系数（相对于最大响应）

        Returns:
            np.ndarray: 检测到的点

        应用场景：
            - 噪声点检测
            - 孤立特征点检测
        """
        # 拉普拉斯算子（突出中心点）
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=np.float32)

        # 卷积
        response = cv2.filter2D(image.astype(np.float32), -1, kernel)

        # 取绝对值
        response = np.abs(response)

        # 阈值化
        max_response = np.max(response)
        threshold_value = threshold * max_response
        points = (response > threshold_value).astype(np.uint8) * 255

        return points

    # ==================== 线检测 ====================

    def line_detection(self, image: np.ndarray,
                      direction: str = 'horizontal',
                      threshold: float = 0.5) -> np.ndarray:
        """
        线检测

        检测特定方向的线条。

        Args:
            image: 输入灰度图像
            direction: 线条方向
                'horizontal' - 水平线
                'vertical' - 垂直线
                'diagonal_45' - 45度对角线
                'diagonal_135' - 135度对角线
            threshold: 阈值系数

        Returns:
            np.ndarray: 检测到的线条

        应用场景：
            - 文档中的线条检测
            - 结构分析
        """
        # 不同方向的线检测算子
        if direction == 'horizontal':
            kernel = np.array([[-1, -1, -1],
                             [ 2,  2,  2],
                             [-1, -1, -1]], dtype=np.float32)
        elif direction == 'vertical':
            kernel = np.array([[-1,  2, -1],
                             [-1,  2, -1],
                             [-1,  2, -1]], dtype=np.float32)
        elif direction == 'diagonal_45':
            kernel = np.array([[-1, -1,  2],
                             [-1,  2, -1],
                             [ 2, -1, -1]], dtype=np.float32)
        elif direction == 'diagonal_135':
            kernel = np.array([[ 2, -1, -1],
                             [-1,  2, -1],
                             [-1, -1,  2]], dtype=np.float32)
        else:
            raise ValueError(f"不支持的方向: {direction}")

        # 卷积
        response = cv2.filter2D(image.astype(np.float32), -1, kernel)

        # 取绝对值
        response = np.abs(response)

        # 阈值化
        max_response = np.max(response)
        threshold_value = threshold * max_response
        lines = (response > threshold_value).astype(np.uint8) * 255

        return lines

    # ==================== Roberts算子 ====================

    def roberts_edge(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Roberts边缘检测

        使用2×2算子，对角线方向梯度。

        Gx = [+1  0]    Gy = [ 0 +1]
             [ 0 -1]         [-1  0]

        Args:
            image: 输入灰度图像

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - 梯度幅值
                - X方向梯度
                - Y方向梯度

        特点：
            - 最简单的边缘检测算子
            - 对噪声敏感
            - 边缘定位准确
            - 对角线方向敏感

        应用场景：
            - 快速边缘检测
            - 噪声较小的图像
        """
        # Roberts算子
        kernel_x = np.array([[1,  0],
                            [0, -1]], dtype=np.float32)

        kernel_y = np.array([[0,  1],
                            [-1, 0]], dtype=np.float32)

        # 计算梯度
        img_float = image.astype(np.float32)
        gx = cv2.filter2D(img_float, -1, kernel_x)
        gy = cv2.filter2D(img_float, -1, kernel_y)

        # 梯度幅值
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

        return magnitude, gx, gy

    # ==================== Prewitt算子 ====================

    def prewitt_edge(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prewitt边缘检测

        使用3×3算子，强调边缘检测的平滑性。

        Gx = [-1  0 +1]    Gy = [-1 -1 -1]
             [-1  0 +1]         [ 0  0  0]
             [-1  0 +1]         [+1 +1 +1]

        Args:
            image: 输入灰度图像

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - 梯度幅值
                - X方向梯度
                - Y方向梯度

        特点：
            - 3×3算子
            - 简单均值滤波
            - 对噪声有一定抑制
            - 边缘较粗

        应用场景：
            - 一般边缘检测
            - 对噪声有一定容忍度
        """
        # Prewitt算子
        kernel_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]], dtype=np.float32)

        kernel_y = np.array([[-1, -1, -1],
                            [ 0,  0,  0],
                            [ 1,  1,  1]], dtype=np.float32)

        # 计算梯度
        img_float = image.astype(np.float32)
        gx = cv2.filter2D(img_float, -1, kernel_x)
        gy = cv2.filter2D(img_float, -1, kernel_y)

        # 梯度幅值
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

        return magnitude, gx, gy

    # ==================== Sobel算子 ====================

    def sobel_edge(self, image: np.ndarray,
                   ksize: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sobel边缘检测

        使用高斯平滑的导数算子，噪声抑制最好。

        3×3 Sobel:
        Gx = [-1  0 +1]    Gy = [-1 -2 -1]
             [-2  0 +2]         [ 0  0  0]
             [-1  0 +1]         [+1 +2 +1]

        Args:
            image: 输入灰度图像
            ksize: 核大小（1, 3, 5, 7）

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - 梯度幅值
                - X方向梯度
                - Y方向梯度

        特点：
            - 高斯加权
            - 噪声抑制最好
            - 边缘定位准确
            - 最常用的边缘检测算子

        应用场景：
            - 标准边缘检测
            - 噪声图像
            - 工业应用
        """
        # 使用OpenCV的Sobel函数
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize)

        # 梯度幅值
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

        return magnitude, gx, gy

    # ==================== Canny边缘检测 ====================

    def canny_edge(self, image: np.ndarray,
                   low_threshold: int = 50,
                   high_threshold: int = 150,
                   aperture_size: int = 3,
                   L2gradient: bool = False) -> np.ndarray:
        """
        Canny边缘检测

        最优边缘检测算法，包含多个步骤：
        1. 高斯滤波去噪
        2. 计算梯度幅值和方向
        3. 非极大值抑制
        4. 双阈值检测
        5. 边缘跟踪

        Args:
            image: 输入灰度图像
            low_threshold: 低阈值
            high_threshold: 高阈值
            aperture_size: Sobel算子大小
            L2gradient: 是否使用L2范数（更准确但更慢）

        Returns:
            np.ndarray: 边缘图像

        特点：
            - 最优边缘检测
            - 边缘连续
            - 单像素宽度
            - 噪声抑制好

        应用场景：
            - 高质量边缘检测
            - 物体检测
            - 图像分割
        """
        edges = cv2.Canny(image, low_threshold, high_threshold,
                         apertureSize=aperture_size, L2gradient=L2gradient)
        return edges

    # ==================== 辅助方法 ====================

    def load_image(self, path: str) -> bool:
        """加载灰度图像"""
        self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return self.image is not None

    def compare_edge_detectors(self, image: np.ndarray,
                               save_path: Optional[str] = None):
        """比较不同边缘检测算子"""
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # 原图
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')

        # Roberts
        roberts, _, _ = self.roberts_edge(image)
        axes[1].imshow(roberts, cmap='gray')
        axes[1].set_title('Roberts边缘检测', fontsize=12)
        axes[1].axis('off')

        # Prewitt
        prewitt, _, _ = self.prewitt_edge(image)
        axes[2].imshow(prewitt, cmap='gray')
        axes[2].set_title('Prewitt边缘检测', fontsize=12)
        axes[2].axis('off')

        # Sobel 3×3
        sobel3, _, _ = self.sobel_edge(image, ksize=3)
        axes[3].imshow(sobel3, cmap='gray')
        axes[3].set_title('Sobel边缘检测 (3×3)', fontsize=12)
        axes[3].axis('off')

        # Sobel 5×5
        sobel5, _, _ = self.sobel_edge(image, ksize=5)
        axes[4].imshow(sobel5, cmap='gray')
        axes[4].set_title('Sobel边缘检测 (5×5)', fontsize=12)
        axes[4].axis('off')

        # Canny
        canny = self.canny_edge(image, 50, 150)
        axes[5].imshow(canny, cmap='gray')
        axes[5].set_title('Canny边缘检测', fontsize=12)
        axes[5].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"边缘检测对比图已保存到: {save_path}")

        plt.show()

    def demonstrate_point_line_detection(self, image: np.ndarray,
                                        save_path: Optional[str] = None):
        """演示点检测和线检测"""
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # 原图
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')

        # 点检测
        points = self.point_detection(image, threshold=0.5)
        axes[1].imshow(points, cmap='gray')
        axes[1].set_title('点检测', fontsize=12)
        axes[1].axis('off')

        # 水平线检测
        h_lines = self.line_detection(image, 'horizontal', threshold=0.5)
        axes[2].imshow(h_lines, cmap='gray')
        axes[2].set_title('水平线检测', fontsize=12)
        axes[2].axis('off')

        # 垂直线检测
        v_lines = self.line_detection(image, 'vertical', threshold=0.5)
        axes[3].imshow(v_lines, cmap='gray')
        axes[3].set_title('垂直线检测', fontsize=12)
        axes[3].axis('off')

        # 45度对角线检测
        d45_lines = self.line_detection(image, 'diagonal_45', threshold=0.5)
        axes[4].imshow(d45_lines, cmap='gray')
        axes[4].set_title('45°对角线检测', fontsize=12)
        axes[4].axis('off')

        # 135度对角线检测
        d135_lines = self.line_detection(image, 'diagonal_135', threshold=0.5)
        axes[5].imshow(d135_lines, cmap='gray')
        axes[5].set_title('135°对角线检测', fontsize=12)
        axes[5].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"点线检测对比图已保存到: {save_path}")

        plt.show()


def main():
    """演示程序"""
    print("边缘检测演示程序")
    print("=" * 50)

    detector = EdgeDetection()

    # 创建测试图像
    print("\n创建测试图像...")
    test_image = np.zeros((256, 256), dtype=np.uint8)

    # 绘制一些形状
    cv2.rectangle(test_image, (50, 50), (150, 150), 200, 3)
    cv2.circle(test_image, (200, 100), 40, 200, 3)
    cv2.line(test_image, (50, 200), (200, 200), 200, 3)

    cv2.imwrite('edge_test_image.png', test_image)

    # 测试1: 边缘检测算子对比
    print("\n测试1: 边缘检测算子对比")
    detector.compare_edge_detectors(test_image, save_path='edge_comparison.png')

    # 测试2: 点检测和线检测
    print("\n测试2: 点检测和线检测")
    detector.demonstrate_point_line_detection(test_image,
                                             save_path='point_line_detection.png')

    # 测试3: 梯度方向
    print("\n测试3: 梯度方向可视化")
    sobel_mag, gx, gy = detector.sobel_edge(test_image)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('原始图像', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(gx, cmap='gray')
    axes[1].set_title('X方向梯度 (Gx)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(gy, cmap='gray')
    axes[2].set_title('Y方向梯度 (Gy)', fontsize=12)
    axes[2].axis('off')

    axes[3].imshow(sobel_mag, cmap='gray')
    axes[3].set_title('梯度幅值', fontsize=12)
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('gradient_visualization.png', dpi=150, bbox_inches='tight')
    print("梯度可视化已保存到: gradient_visualization.png")
    plt.show()

    print("\n✅ 演示完成！")
    print(f"\n生成文件:")
    print(f"  - edge_test_image.png (测试图像)")
    print(f"  - edge_comparison.png (边缘检测对比)")
    print(f"  - point_line_detection.png (点线检测)")
    print(f"  - gradient_visualization.png (梯度可视化)")


if __name__ == '__main__':
    main()
