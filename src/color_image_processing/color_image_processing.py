"""
彩色图像平滑和锐化模块

实现彩色图像的平滑和锐化处理：
- 平滑：高斯平滑、双边滤波、均值平滑
- 锐化：拉普拉斯锐化、Unsharp Masking、高通滤波
- 支持多种颜色空间：RGB、HSV、LAB
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Literal
import matplotlib.pyplot as plt


class ColorImageProcessing:
    """彩色图像平滑和锐化类"""

    def __init__(self):
        self.image = None

    # ==================== 平滑处理 ====================

    def gaussian_smooth(self, image: np.ndarray,
                       kernel_size: int = 5,
                       sigma: float = 1.0,
                       color_space: str = 'RGB') -> np.ndarray:
        """
        高斯平滑

        Args:
            image: 输入彩色图像 (BGR格式)
            kernel_size: 卷积核大小（奇数）
            sigma: 高斯标准差
            color_space: 颜色空间 'RGB', 'HSV', 'LAB'

        Returns:
            np.ndarray: 平滑后的图像

        应用场景：
            - 去除高斯噪声
            - 图像预处理
            - 减少细节
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        if color_space == 'RGB':
            # 直接对BGR三通道分别处理
            result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        elif color_space == 'HSV':
            # 转换到HSV，只对V通道平滑
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.GaussianBlur(hsv[:, :, 2], (kernel_size, kernel_size), sigma)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif color_space == 'LAB':
            # 转换到LAB，只对L通道平滑
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.GaussianBlur(lab[:, :, 0], (kernel_size, kernel_size), sigma)
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        else:
            raise ValueError(f"不支持的颜色空间: {color_space}")

        return result

    def bilateral_filter(self, image: np.ndarray,
                        d: int = 9,
                        sigma_color: float = 75,
                        sigma_space: float = 75) -> np.ndarray:
        """
        双边滤波（保边平滑）

        既考虑空间距离，又考虑像素值差异，能很好地保持边缘。

        Args:
            image: 输入彩色图像
            d: 邻域直径
            sigma_color: 颜色空间标准差
            sigma_space: 坐标空间标准差

        Returns:
            np.ndarray: 滤波后的图像

        应用场景：
            - 保边平滑
            - 美颜效果
            - 去噪同时保持边缘
        """
        result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return result

    def mean_smooth(self, image: np.ndarray,
                   kernel_size: int = 5,
                   color_space: str = 'RGB') -> np.ndarray:
        """
        均值平滑

        Args:
            image: 输入彩色图像
            kernel_size: 卷积核大小
            color_space: 颜色空间

        Returns:
            np.ndarray: 平滑后的图像
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        if color_space == 'RGB':
            result = cv2.blur(image, (kernel_size, kernel_size))

        elif color_space == 'HSV':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.blur(hsv[:, :, 2], (kernel_size, kernel_size))
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif color_space == 'LAB':
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.blur(lab[:, :, 0], (kernel_size, kernel_size))
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        else:
            raise ValueError(f"不支持的颜色空间: {color_space}")

        return result

    def median_smooth(self, image: np.ndarray,
                     kernel_size: int = 5) -> np.ndarray:
        """
        中值平滑（对彩色图像）

        Args:
            image: 输入彩色图像
            kernel_size: 卷积核大小

        Returns:
            np.ndarray: 平滑后的图像
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        # 对每个通道分别进行中值滤波
        result = cv2.medianBlur(image, kernel_size)
        return result

    # ==================== 锐化处理 ====================

    def laplacian_sharpen(self, image: np.ndarray,
                         kernel_type: str = '4',
                         alpha: float = 1.0,
                         color_space: str = 'RGB') -> np.ndarray:
        """
        拉普拉斯锐化

        公式: sharpened = original - alpha × laplacian

        Args:
            image: 输入彩色图像
            kernel_type: 拉普拉斯核类型 '4', '8', 'enhanced'
            alpha: 锐化强度
            color_space: 颜色空间

        Returns:
            np.ndarray: 锐化后的图像

        应用场景：
            - 增强边缘
            - 提高清晰度
            - 突出细节
        """
        # 拉普拉斯核
        if kernel_type == '4':
            kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float32)
        elif kernel_type == '8':
            kernel = np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]], dtype=np.float32)
        elif kernel_type == 'enhanced':
            kernel = np.array([[1, 4, 1],
                             [4, -20, 4],
                             [1, 4, 1]], dtype=np.float32)
        else:
            raise ValueError(f"不支持的kernel_type: {kernel_type}")

        if color_space == 'RGB':
            # 对每个通道分别处理
            img_float = image.astype(np.float32)
            laplacian = cv2.filter2D(img_float, -1, kernel)
            sharpened = img_float - alpha * laplacian

        elif color_space == 'HSV':
            # 只对V通道锐化
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_float = hsv[:, :, 2].astype(np.float32)
            laplacian = cv2.filter2D(v_float, -1, kernel)
            hsv[:, :, 2] = np.clip(v_float - alpha * laplacian, 0, 255).astype(np.uint8)
            sharpened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

        elif color_space == 'LAB':
            # 只对L通道锐化
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_float = lab[:, :, 0].astype(np.float32)
            laplacian = cv2.filter2D(l_float, -1, kernel)
            lab[:, :, 0] = np.clip(l_float - alpha * laplacian, 0, 255).astype(np.uint8)
            sharpened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

        else:
            raise ValueError(f"不支持的颜色空间: {color_space}")

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def unsharp_masking(self, image: np.ndarray,
                       kernel_size: int = 5,
                       sigma: float = 1.0,
                       amount: float = 1.5,
                       threshold: int = 0,
                       color_space: str = 'RGB') -> np.ndarray:
        """
        Unsharp Masking 锐化

        公式: sharpened = original + amount × (original - blurred)

        Args:
            image: 输入彩色图像
            kernel_size: 高斯核大小
            sigma: 高斯标准差
            amount: 锐化强度（通常1.0-2.0）
            threshold: 阈值（只锐化差异大于此值的像素）
            color_space: 颜色空间

        Returns:
            np.ndarray: 锐化后的图像

        应用场景：
            - 照片锐化
            - 打印前处理
            - 细节增强
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        if color_space == 'RGB':
            # 对整个图像处理
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            mask = cv2.subtract(image, blurred)

            if threshold > 0:
                # 只在差异大于阈值的地方锐化
                mask = np.where(np.abs(mask) < threshold, 0, mask)

            sharpened = cv2.addWeighted(image, 1.0, mask, amount, 0)

        elif color_space == 'HSV':
            # 只对V通道处理
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]

            blurred = cv2.GaussianBlur(v_channel, (kernel_size, kernel_size), sigma)
            mask = cv2.subtract(v_channel, blurred)

            if threshold > 0:
                mask = np.where(np.abs(mask) < threshold, 0, mask)

            hsv[:, :, 2] = cv2.addWeighted(v_channel, 1.0, mask, amount, 0)
            sharpened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif color_space == 'LAB':
            # 只对L通道处理
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]

            blurred = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), sigma)
            mask = cv2.subtract(l_channel, blurred)

            if threshold > 0:
                mask = np.where(np.abs(mask) < threshold, 0, mask)

            lab[:, :, 0] = cv2.addWeighted(l_channel, 1.0, mask, amount, 0)
            sharpened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        else:
            raise ValueError(f"不支持的颜色空间: {color_space}")

        return sharpened

    def highboost_filter(self, image: np.ndarray,
                        kernel_size: int = 5,
                        A: float = 1.5,
                        color_space: str = 'RGB') -> np.ndarray:
        """
        高提升滤波

        公式: sharpened = A × original - blurred

        Args:
            image: 输入彩色图像
            kernel_size: 平滑核大小
            A: 提升系数（A > 1）
            color_space: 颜色空间

        Returns:
            np.ndarray: 锐化后的图像
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        if A <= 1:
            raise ValueError("A必须大于1")

        if color_space == 'RGB':
            blurred = cv2.blur(image, (kernel_size, kernel_size))
            sharpened = cv2.addWeighted(image, A, blurred, -1.0, 0)

        elif color_space == 'HSV':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            blurred = cv2.blur(v_channel, (kernel_size, kernel_size))
            hsv[:, :, 2] = cv2.addWeighted(v_channel, A, blurred, -1.0, 0)
            sharpened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif color_space == 'LAB':
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            blurred = cv2.blur(l_channel, (kernel_size, kernel_size))
            lab[:, :, 0] = cv2.addWeighted(l_channel, A, blurred, -1.0, 0)
            sharpened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        else:
            raise ValueError(f"不支持的颜色空间: {color_space}")

        return sharpened

    # ==================== 辅助方法 ====================

    def load_image(self, path: str) -> bool:
        """加载彩色图像"""
        self.image = cv2.imread(path, cv2.IMREAD_COLOR)
        return self.image is not None

    def compare_smoothing(self, image: np.ndarray,
                         save_path: Optional[str] = None):
        """比较不同平滑方法"""
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # 原图
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')

        # 高斯平滑 RGB
        gauss_rgb = self.gaussian_smooth(image, kernel_size=5, color_space='RGB')
        axes[1].imshow(cv2.cvtColor(gauss_rgb, cv2.COLOR_BGR2RGB))
        axes[1].set_title('高斯平滑 (RGB)', fontsize=12)
        axes[1].axis('off')

        # 高斯平滑 HSV
        gauss_hsv = self.gaussian_smooth(image, kernel_size=5, color_space='HSV')
        axes[2].imshow(cv2.cvtColor(gauss_hsv, cv2.COLOR_BGR2RGB))
        axes[2].set_title('高斯平滑 (HSV-V通道)', fontsize=12)
        axes[2].axis('off')

        # 双边滤波
        bilateral = self.bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)
        axes[3].imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB))
        axes[3].set_title('双边滤波 (保边)', fontsize=12)
        axes[3].axis('off')

        # 均值平滑
        mean = self.mean_smooth(image, kernel_size=5)
        axes[4].imshow(cv2.cvtColor(mean, cv2.COLOR_BGR2RGB))
        axes[4].set_title('均值平滑', fontsize=12)
        axes[4].axis('off')

        # 中值平滑
        median = self.median_smooth(image, kernel_size=5)
        axes[5].imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
        axes[5].set_title('中值平滑', fontsize=12)
        axes[5].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"平滑对比图已保存到: {save_path}")

        plt.show()

    def compare_sharpening(self, image: np.ndarray,
                          save_path: Optional[str] = None):
        """比较不同锐化方法"""
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # 原图
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')

        # 拉普拉斯锐化 RGB
        lap_rgb = self.laplacian_sharpen(image, kernel_type='4', alpha=1.0, color_space='RGB')
        axes[1].imshow(cv2.cvtColor(lap_rgb, cv2.COLOR_BGR2RGB))
        axes[1].set_title('拉普拉斯锐化 (RGB)', fontsize=12)
        axes[1].axis('off')

        # 拉普拉斯锐化 HSV
        lap_hsv = self.laplacian_sharpen(image, kernel_type='4', alpha=1.0, color_space='HSV')
        axes[2].imshow(cv2.cvtColor(lap_hsv, cv2.COLOR_BGR2RGB))
        axes[2].set_title('拉普拉斯锐化 (HSV-V通道)', fontsize=12)
        axes[2].axis('off')

        # Unsharp Masking RGB
        unsharp_rgb = self.unsharp_masking(image, amount=1.5, color_space='RGB')
        axes[3].imshow(cv2.cvtColor(unsharp_rgb, cv2.COLOR_BGR2RGB))
        axes[3].set_title('Unsharp Masking (RGB)', fontsize=12)
        axes[3].axis('off')

        # Unsharp Masking LAB
        unsharp_lab = self.unsharp_masking(image, amount=1.5, color_space='LAB')
        axes[4].imshow(cv2.cvtColor(unsharp_lab, cv2.COLOR_BGR2RGB))
        axes[4].set_title('Unsharp Masking (LAB-L通道)', fontsize=12)
        axes[4].axis('off')

        # 高提升滤波
        highboost = self.highboost_filter(image, A=1.5, color_space='RGB')
        axes[5].imshow(cv2.cvtColor(highboost, cv2.COLOR_BGR2RGB))
        axes[5].set_title('高提升滤波', fontsize=12)
        axes[5].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"锐化对比图已保存到: {save_path}")

        plt.show()


def main():
    """演示程序"""
    print("彩色图像平滑和锐化演示程序")
    print("=" * 50)

    processor = ColorImageProcessing()

    # 创建测试彩色图像
    print("\n创建测试图像...")
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)

    # 绘制彩色图形
    cv2.circle(test_image, (128, 128), 80, (0, 0, 255), -1)  # 红色圆
    cv2.rectangle(test_image, (50, 50), (100, 100), (0, 255, 0), -1)  # 绿色矩形
    cv2.rectangle(test_image, (150, 150), (200, 200), (255, 0, 0), -1)  # 蓝色矩形

    cv2.imwrite('color_test_image.png', test_image)

    # 测试1：平滑处理
    print("\n测试1: 平滑处理对比")
    processor.compare_smoothing(test_image, save_path='color_smoothing_comparison.png')

    # 测试2：锐化处理
    print("\n测试2: 锐化处理对比")
    # 先平滑一下，让锐化效果更明显
    blurred = processor.gaussian_smooth(test_image, kernel_size=5)
    processor.compare_sharpening(blurred, save_path='color_sharpening_comparison.png')

    # 测试3：添加噪声后的平滑
    print("\n测试3: 噪声图像平滑")
    from noise_simulation import NoiseSimulation
    noise_sim = NoiseSimulation()

    # 对每个通道添加高斯噪声
    noisy = test_image.copy().astype(np.float32)
    for i in range(3):
        noise = np.random.normal(0, 15, test_image.shape[:2])
        noisy[:, :, i] += noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    cv2.imwrite('color_noisy_image.png', noisy)
    processor.compare_smoothing(noisy, save_path='color_noisy_smoothing.png')

    print("\n✅ 演示完成！")
    print(f"\n生成文件:")
    print(f"  - color_test_image.png (测试图像)")
    print(f"  - color_smoothing_comparison.png (平滑对比)")
    print(f"  - color_sharpening_comparison.png (锐化对比)")
    print(f"  - color_noisy_image.png (噪声图像)")
    print(f"  - color_noisy_smoothing.png (噪声平滑对比)")


if __name__ == '__main__':
    main()
