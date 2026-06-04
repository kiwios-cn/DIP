"""
均值滤波器模块

实现四种均值滤波器：
1. 算术均值滤波器 (Arithmetic Mean Filter)
2. 几何均值滤波器 (Geometric Mean Filter)
3. 谐波均值滤波器 (Harmonic Mean Filter)
4. 逆谐波均值滤波器 (Contraharmonic Mean Filter)
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class MeanFilters:
    """均值滤波器类"""

    def __init__(self):
        self.image = None

    # ==================== 核心滤波器 ====================

    def arithmetic_mean_filter(self, image: np.ndarray,
                               kernel_size: int = 3) -> np.ndarray:
        """
        算术均值滤波器

        对邻域内所有像素求算术平均值。适合去除高斯噪声。

        公式: f_hat(x,y) = (1/mn) * Σ g(s,t)

        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数），默认3

        Returns:
            np.ndarray: 滤波后的图像

        应用场景：
            - 去除高斯噪声
            - 图像平滑
            - 预处理

        优点：
            - 简单高效
            - 对高斯噪声效果好

        缺点：
            - 会模糊图像边缘
            - 对椒盐噪声效果差
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        # 使用OpenCV的均值滤波（高效）
        result = cv2.blur(image, (kernel_size, kernel_size))
        return result

    def geometric_mean_filter(self, image: np.ndarray,
                              kernel_size: int = 3) -> np.ndarray:
        """
        几何均值滤波器

        对邻域内所有像素求几何平均值。平滑效果介于算术均值和中值之间。

        公式: f_hat(x,y) = [Π g(s,t)]^(1/mn)

        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数），默认3

        Returns:
            np.ndarray: 滤波后的图像

        应用场景：
            - 去除高斯噪声
            - 保持边缘细节
            - 图像平滑

        优点：
            - 比算术均值更好地保持边缘
            - 对高斯噪声效果好

        缺点：
            - 计算复杂度较高
            - 遇到0值会出问题（需要特殊处理）
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        # 转换为float，避免溢出
        img_float = image.astype(np.float64) + 1e-10  # 避免log(0)

        # 使用对数运算避免数值溢出
        # geometric_mean = exp(mean(log(x)))
        log_img = np.log(img_float)

        # 卷积核
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
        kernel = kernel / kernel.sum()

        # 使用filter2D进行卷积
        log_filtered = cv2.filter2D(log_img, -1, kernel)

        # 转换回原始空间
        result = np.exp(log_filtered)

        return np.clip(result, 0, 255).astype(np.uint8)

    def harmonic_mean_filter(self, image: np.ndarray,
                            kernel_size: int = 3) -> np.ndarray:
        """
        谐波均值滤波器

        对邻域内所有像素求谐波平均值。适合处理盐噪声（亮点）。

        公式: f_hat(x,y) = mn / Σ(1/g(s,t))

        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数），默认3

        Returns:
            np.ndarray: 滤波后的图像

        应用场景：
            - 去除盐噪声（白点）
            - 处理高亮异常点
            - 图像增强

        优点：
            - 对盐噪声（亮点）效果好
            - 能保持暗区域细节

        缺点：
            - 对椒噪声（黑点）效果差
            - 遇到0值会出问题
            - 会使图像变暗
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        # 转换为float并避免除零
        img_float = image.astype(np.float64)
        img_float[img_float == 0] = 1e-10  # 避免除零

        # 计算倒数
        reciprocal = 1.0 / img_float

        # 卷积核
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)

        # 对倒数求和
        sum_reciprocal = cv2.filter2D(reciprocal, -1, kernel)

        # 谐波均值 = n / sum(1/x)
        n = kernel_size * kernel_size
        result = n / (sum_reciprocal + 1e-10)  # 避免除零

        return np.clip(result, 0, 255).astype(np.uint8)

    def contraharmonic_mean_filter(self, image: np.ndarray,
                                   kernel_size: int = 3,
                                   Q: float = 1.0) -> np.ndarray:
        """
        逆谐波均值滤波器

        对邻域内像素进行加权平均，权重由阶数Q控制。

        公式: f_hat(x,y) = Σ g(s,t)^(Q+1) / Σ g(s,t)^Q

        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数），默认3
            Q: 阶数（order）
                Q > 0: 去除椒噪声（黑点）
                Q < 0: 去除盐噪声（白点）
                Q = 0: 等同于算术均值
                Q = -1: 等同于谐波均值

        Returns:
            np.ndarray: 滤波后的图像

        应用场景：
            - Q > 0: 去除椒噪声（黑点）
            - Q < 0: 去除盐噪声（白点）
            - Q = 1.5: 强力去除黑点
            - Q = -1.5: 强力去除白点

        优点：
            - 通过调整Q可以针对性去除椒噪声或盐噪声
            - 灵活性高

        缺点：
            - Q选择不当会产生负面效果
            - 计算复杂度较高
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        # 转换为float，避免溢出
        img_float = image.astype(np.float64) + 1e-10  # 避免0值

        # 计算分子：g^(Q+1)
        numerator = np.power(img_float, Q + 1)

        # 计算分母：g^Q
        denominator = np.power(img_float, Q)

        # 卷积核
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)

        # 对分子分母分别求和
        sum_numerator = cv2.filter2D(numerator, -1, kernel)
        sum_denominator = cv2.filter2D(denominator, -1, kernel)

        # 逆谐波均值
        result = sum_numerator / (sum_denominator + 1e-10)

        return np.clip(result, 0, 255).astype(np.uint8)

    # ==================== 辅助方法 ====================

    def load_image(self, path: str, grayscale: bool = True) -> bool:
        """
        加载图像

        Args:
            path: 图像路径
            grayscale: 是否转为灰度图

        Returns:
            bool: 是否成功加载
        """
        if grayscale:
            self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            self.image = cv2.imread(path)

        return self.image is not None

    def compare_filters(self, image: np.ndarray,
                       kernel_size: int = 3,
                       Q_positive: float = 1.5,
                       Q_negative: float = -1.5,
                       save_path: Optional[str] = None):
        """
        比较所有滤波器效果

        Args:
            image: 输入图像
            kernel_size: 卷积核大小
            Q_positive: 逆谐波滤波器正阶数
            Q_negative: 逆谐波滤波器负阶数
            save_path: 保存路径
        """
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # 原图
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')

        # 算术均值
        arith = self.arithmetic_mean_filter(image, kernel_size)
        axes[1].imshow(arith, cmap='gray')
        axes[1].set_title(f'算术均值滤波\n(kernel={kernel_size}×{kernel_size})', fontsize=12)
        axes[1].axis('off')

        # 几何均值
        geom = self.geometric_mean_filter(image, kernel_size)
        axes[2].imshow(geom, cmap='gray')
        axes[2].set_title(f'几何均值滤波\n(kernel={kernel_size}×{kernel_size})', fontsize=12)
        axes[2].axis('off')

        # 谐波均值
        harm = self.harmonic_mean_filter(image, kernel_size)
        axes[3].imshow(harm, cmap='gray')
        axes[3].set_title(f'谐波均值滤波\n(kernel={kernel_size}×{kernel_size})', fontsize=12)
        axes[3].axis('off')

        # 逆谐波均值 (Q > 0, 去除椒噪声)
        contra_pos = self.contraharmonic_mean_filter(image, kernel_size, Q_positive)
        axes[4].imshow(contra_pos, cmap='gray')
        axes[4].set_title(f'逆谐波均值滤波\n(Q={Q_positive}, 去除椒噪声)', fontsize=12)
        axes[4].axis('off')

        # 逆谐波均值 (Q < 0, 去除盐噪声)
        contra_neg = self.contraharmonic_mean_filter(image, kernel_size, Q_negative)
        axes[5].imshow(contra_neg, cmap='gray')
        axes[5].set_title(f'逆谐波均值滤波\n(Q={Q_negative}, 去除盐噪声)', fontsize=12)
        axes[5].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"对比图已保存到: {save_path}")

        plt.show()

    def demonstrate_Q_effect(self, image: np.ndarray,
                            kernel_size: int = 3,
                            save_path: Optional[str] = None):
        """
        演示逆谐波滤波器不同Q值的效果

        Args:
            image: 输入图像
            kernel_size: 卷积核大小
            save_path: 保存路径
        """
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        Q_values = [-2, -1, 0, 1, 2]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # 原图
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')

        # 不同Q值
        for idx, Q in enumerate(Q_values, start=1):
            filtered = self.contraharmonic_mean_filter(image, kernel_size, Q)
            axes[idx].imshow(filtered, cmap='gray')

            if Q > 0:
                effect = "去除椒噪声"
            elif Q < 0:
                effect = "去除盐噪声"
            else:
                effect = "算术均值"

            axes[idx].set_title(f'Q = {Q}\n({effect})', fontsize=12)
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Q值效果对比图已保存到: {save_path}")

        plt.show()


def main():
    """演示程序"""
    print("均值滤波器演示程序")
    print("=" * 50)

    # 创建滤波器对象
    filters = MeanFilters()

    # 创建测试图像
    test_image = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(test_image, (128, 128), 80, 200, -1)
    cv2.rectangle(test_image, (50, 50), (100, 100), 150, -1)

    # 添加高斯噪声
    from noise_simulation import NoiseSimulation
    noise_sim = NoiseSimulation()

    # 测试1：高斯噪声
    print("\n测试1: 高斯噪声去除")
    gaussian_noisy = noise_sim.add_gaussian_noise(test_image, sigma=25)
    cv2.imwrite('mean_filters_gaussian_noisy.png', gaussian_noisy)

    filters.compare_filters(
        gaussian_noisy,
        kernel_size=5,
        save_path='mean_filters_gaussian_comparison.png'
    )

    # 测试2：椒盐噪声
    print("\n测试2: 椒盐噪声去除")
    sp_noisy = noise_sim.add_salt_pepper_noise(test_image, prob=0.1)
    cv2.imwrite('mean_filters_sp_noisy.png', sp_noisy)

    filters.compare_filters(
        sp_noisy,
        kernel_size=5,
        Q_positive=1.5,
        Q_negative=-1.5,
        save_path='mean_filters_sp_comparison.png'
    )

    # 测试3：只有盐噪声
    print("\n测试3: 盐噪声去除（只有白点）")
    salt_noisy = noise_sim.add_salt_pepper_noise(test_image, prob=0.1, salt_prob=1.0)
    cv2.imwrite('mean_filters_salt_noisy.png', salt_noisy)

    filters.compare_filters(
        salt_noisy,
        kernel_size=5,
        Q_negative=-1.5,
        save_path='mean_filters_salt_comparison.png'
    )

    # 测试4：只有椒噪声
    print("\n测试4: 椒噪声去除（只有黑点）")
    pepper_noisy = noise_sim.add_salt_pepper_noise(test_image, prob=0.1, salt_prob=0.0)
    cv2.imwrite('mean_filters_pepper_noisy.png', pepper_noisy)

    filters.compare_filters(
        pepper_noisy,
        kernel_size=5,
        Q_positive=1.5,
        save_path='mean_filters_pepper_comparison.png'
    )

    # 测试5：Q值效果演示
    print("\n测试5: 逆谐波滤波器Q值效果")
    filters.demonstrate_Q_effect(
        sp_noisy,
        kernel_size=5,
        save_path='mean_filters_Q_effect.png'
    )

    print("\n✅ 演示完成！")
    print(f"\n生成文件:")
    print(f"  - mean_filters_gaussian_noisy.png (高斯噪声图像)")
    print(f"  - mean_filters_gaussian_comparison.png (高斯噪声滤波对比)")
    print(f"  - mean_filters_sp_noisy.png (椒盐噪声图像)")
    print(f"  - mean_filters_sp_comparison.png (椒盐噪声滤波对比)")
    print(f"  - mean_filters_salt_noisy.png (盐噪声图像)")
    print(f"  - mean_filters_salt_comparison.png (盐噪声滤波对比)")
    print(f"  - mean_filters_pepper_noisy.png (椒噪声图像)")
    print(f"  - mean_filters_pepper_comparison.png (椒噪声滤波对比)")
    print(f"  - mean_filters_Q_effect.png (Q值效果对比)")


if __name__ == '__main__':
    main()
