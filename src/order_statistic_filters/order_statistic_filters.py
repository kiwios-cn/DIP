"""
统计排序滤波器模块

实现基于排序统计的非线性滤波器：
1. 中值滤波器 (Median Filter)
2. 最大值滤波器 (Max Filter)
3. 最小值滤波器 (Min Filter)
4. 中点滤波器 (Midpoint Filter)
5. 修正阿尔法均值滤波器 (Alpha-Trimmed Mean Filter)
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class OrderStatisticFilters:
    """统计排序滤波器类"""

    def __init__(self):
        self.image = None

    # ==================== 核心滤波器 ====================

    def median_filter(self, image: np.ndarray,
                     kernel_size: int = 3) -> np.ndarray:
        """
        中值滤波器

        用邻域像素的中值替代中心像素。对椒盐噪声效果最好。

        公式: f_hat(x,y) = median{g(s,t)}

        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数），默认3

        Returns:
            np.ndarray: 滤波后的图像

        应用场景：
            - 去除椒盐噪声（最佳选择）
            - 保持边缘
            - 去除孤立噪声点

        优点：
            - 对椒盐噪声效果极好
            - 能保持边缘
            - 不会产生新的像素值

        缺点：
            - 计算复杂度高（需要排序）
            - 对高斯噪声效果一般
            - 会使图像略微模糊
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        # 使用OpenCV的中值滤波（高效）
        result = cv2.medianBlur(image, kernel_size)
        return result

    def max_filter(self, image: np.ndarray,
                   kernel_size: int = 3) -> np.ndarray:
        """
        最大值滤波器

        用邻域像素的最大值替代中心像素。能去除椒噪声（黑点）。

        公式: f_hat(x,y) = max{g(s,t)}

        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数），默认3

        Returns:
            np.ndarray: 滤波后的图像

        应用场景：
            - 去除椒噪声（黑点）
            - 图像膨胀
            - 寻找局部最亮点

        优点：
            - 能有效去除暗点
            - 增强亮区域
            - 计算相对简单

        缺点：
            - 会使图像整体变亮
            - 会扩大亮区域
            - 对盐噪声（白点）会恶化
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        # 创建结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # 使用形态学膨胀实现最大值滤波
        result = cv2.dilate(image, kernel)
        return result

    def min_filter(self, image: np.ndarray,
                   kernel_size: int = 3) -> np.ndarray:
        """
        最小值滤波器

        用邻域像素的最小值替代中心像素。能去除盐噪声（白点）。

        公式: f_hat(x,y) = min{g(s,t)}

        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数），默认3

        Returns:
            np.ndarray: 滤波后的图像

        应用场景：
            - 去除盐噪声（白点）
            - 图像腐蚀
            - 寻找局部最暗点

        优点：
            - 能有效去除亮点
            - 增强暗区域
            - 计算相对简单

        缺点：
            - 会使图像整体变暗
            - 会扩大暗区域
            - 对椒噪声（黑点）会恶化
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        # 创建结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # 使用形态学腐蚀实现最小值滤波
        result = cv2.erode(image, kernel)
        return result

    def midpoint_filter(self, image: np.ndarray,
                       kernel_size: int = 3) -> np.ndarray:
        """
        中点滤波器

        用邻域像素的最大值和最小值的平均值替代中心像素。

        公式: f_hat(x,y) = (max{g(s,t)} + min{g(s,t)}) / 2

        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数），默认3

        Returns:
            np.ndarray: 滤波后的图像

        应用场景：
            - 去除高斯噪声
            - 去除均匀噪声
            - 图像平滑

        优点：
            - 对高斯噪声和均匀噪声效果好
            - 计算相对简单（只需找最大最小值）
            - 比均值滤波更好地保持边缘

        缺点：
            - 对椒盐噪声效果差
            - 受极值影响大
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        # 获取最大值和最小值
        max_img = self.max_filter(image, kernel_size).astype(np.float32)
        min_img = self.min_filter(image, kernel_size).astype(np.float32)

        # 计算中点
        result = (max_img + min_img) / 2.0

        return result.astype(np.uint8)

    def alpha_trimmed_mean_filter(self, image: np.ndarray,
                                  kernel_size: int = 3,
                                  d: int = 2) -> np.ndarray:
        """
        修正阿尔法均值滤波器

        去掉最高和最低的d/2个值后，对剩余像素求均值。

        公式: f_hat(x,y) = (1/(mn-d)) × Σ g_r(s,t)
        其中 g_r 是排序后去除最高和最低d/2个值的像素

        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数），默认3
            d: 去除的像素数量（必须是偶数且 < kernel_size²），默认2

        Returns:
            np.ndarray: 滤波后的图像

        应用场景：
            - 混合噪声（高斯+椒盐）
            - 需要平衡去噪和保边
            - 多种噪声类型共存

        优点：
            - 结合了均值和中值滤波的优点
            - 对多种噪声都有效
            - 通过d调节可以在均值和中值间连续调整

        缺点：
            - 计算复杂度高（需要排序）
            - 参数d需要根据噪声调整

        参数d的选择：
            - d = 0: 等同于算术均值滤波
            - d = kernel_size² - 1: 等同于中值滤波
            - d = 2-4: 轻度修正（去除少数极值）
            - d = kernel_size² / 2: 强修正
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size必须是奇数")

        if d % 2 != 0:
            raise ValueError("d必须是偶数")

        total_pixels = kernel_size * kernel_size
        if d >= total_pixels:
            raise ValueError(f"d必须小于kernel_size² ({total_pixels})")

        # 手动实现滤波（因为需要灵活的排序和裁剪）
        pad_size = kernel_size // 2
        padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size,
                                    cv2.BORDER_REFLECT)

        result = np.zeros_like(image, dtype=np.float32)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # 提取邻域
                neighborhood = padded[i:i+kernel_size, j:j+kernel_size].flatten()

                # 排序
                sorted_values = np.sort(neighborhood)

                # 去除最小的d/2个和最大的d/2个
                d_half = d // 2
                if d_half > 0:
                    trimmed = sorted_values[d_half:-d_half]
                else:
                    trimmed = sorted_values

                # 计算均值
                result[i, j] = np.mean(trimmed)

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
                       kernel_size: int = 5,
                       alpha_d: int = 4,
                       save_path: Optional[str] = None):
        """
        比较所有统计排序滤波器效果

        Args:
            image: 输入图像
            kernel_size: 卷积核大小
            alpha_d: 阿尔法滤波器的d参数
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

        # 中值滤波
        median = self.median_filter(image, kernel_size)
        axes[1].imshow(median, cmap='gray')
        axes[1].set_title(f'中值滤波\n(kernel={kernel_size}×{kernel_size})', fontsize=12)
        axes[1].axis('off')

        # 最大值滤波
        max_f = self.max_filter(image, kernel_size)
        axes[2].imshow(max_f, cmap='gray')
        axes[2].set_title(f'最大值滤波\n(去除黑点)', fontsize=12)
        axes[2].axis('off')

        # 最小值滤波
        min_f = self.min_filter(image, kernel_size)
        axes[3].imshow(min_f, cmap='gray')
        axes[3].set_title(f'最小值滤波\n(去除白点)', fontsize=12)
        axes[3].axis('off')

        # 中点滤波
        midpoint = self.midpoint_filter(image, kernel_size)
        axes[4].imshow(midpoint, cmap='gray')
        axes[4].set_title(f'中点滤波\n(高斯/均匀噪声)', fontsize=12)
        axes[4].axis('off')

        # 修正阿尔法均值滤波
        alpha = self.alpha_trimmed_mean_filter(image, kernel_size, alpha_d)
        axes[5].imshow(alpha, cmap='gray')
        axes[5].set_title(f'修正阿尔法均值滤波\n(d={alpha_d})', fontsize=12)
        axes[5].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"对比图已保存到: {save_path}")

        plt.show()

    def demonstrate_alpha_d_effect(self, image: np.ndarray,
                                   kernel_size: int = 5,
                                   save_path: Optional[str] = None):
        """
        演示修正阿尔法均值滤波器不同d值的效果

        Args:
            image: 输入图像
            kernel_size: 卷积核大小
            save_path: 保存路径
        """
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        total_pixels = kernel_size * kernel_size
        # d必须是偶数且小于total_pixels
        d_values = [0, 2, 4, 8, total_pixels - 1]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # 原图
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')

        # 不同d值
        for idx, d in enumerate(d_values, start=1):
            if d >= total_pixels:
                # 使用中值滤波代替
                filtered = self.median_filter(image, kernel_size)
                effect = "≈ 中值滤波"
            elif d == 0:
                # 等同于算术均值
                filtered = cv2.blur(image, (kernel_size, kernel_size))
                effect = "= 算术均值"
            else:
                filtered = self.alpha_trimmed_mean_filter(image, kernel_size, d)
                percent = (d / total_pixels) * 100
                effect = f"裁剪{percent:.0f}%"

            axes[idx].imshow(filtered, cmap='gray')
            axes[idx].set_title(f'd = {d}\n({effect})', fontsize=12)
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"d值效果对比图已保存到: {save_path}")

        plt.show()


def main():
    """演示程序"""
    print("统计排序滤波器演示程序")
    print("=" * 50)

    # 创建滤波器对象
    filters = OrderStatisticFilters()

    # 创建测试图像
    test_image = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(test_image, (128, 128), 80, 200, -1)
    cv2.rectangle(test_image, (50, 50), (100, 100), 150, -1)

    # 添加噪声
    from noise_simulation import NoiseSimulation
    noise_sim = NoiseSimulation()

    # 测试1：椒盐噪声
    print("\n测试1: 椒盐噪声去除")
    sp_noisy = noise_sim.add_salt_pepper_noise(test_image, prob=0.1)
    cv2.imwrite('order_stat_sp_noisy.png', sp_noisy)

    filters.compare_filters(
        sp_noisy,
        kernel_size=5,
        alpha_d=4,
        save_path='order_stat_sp_comparison.png'
    )

    # 测试2：只有盐噪声
    print("\n测试2: 盐噪声去除（只有白点）")
    salt_noisy = noise_sim.add_salt_pepper_noise(test_image, prob=0.1, salt_prob=1.0)
    cv2.imwrite('order_stat_salt_noisy.png', salt_noisy)

    filters.compare_filters(
        salt_noisy,
        kernel_size=5,
        save_path='order_stat_salt_comparison.png'
    )

    # 测试3：只有椒噪声
    print("\n测试3: 椒噪声去除（只有黑点）")
    pepper_noisy = noise_sim.add_salt_pepper_noise(test_image, prob=0.1, salt_prob=0.0)
    cv2.imwrite('order_stat_pepper_noisy.png', pepper_noisy)

    filters.compare_filters(
        pepper_noisy,
        kernel_size=5,
        save_path='order_stat_pepper_comparison.png'
    )

    # 测试4：高斯噪声
    print("\n测试4: 高斯噪声")
    gaussian_noisy = noise_sim.add_gaussian_noise(test_image, sigma=25)
    cv2.imwrite('order_stat_gaussian_noisy.png', gaussian_noisy)

    filters.compare_filters(
        gaussian_noisy,
        kernel_size=5,
        save_path='order_stat_gaussian_comparison.png'
    )

    # 测试5：混合噪声
    print("\n测试5: 混合噪声（高斯+椒盐）")
    mixed_noisy = noise_sim.add_mixed_noise(test_image, gaussian_sigma=15, sp_prob=0.05)
    cv2.imwrite('order_stat_mixed_noisy.png', mixed_noisy)

    filters.compare_filters(
        mixed_noisy,
        kernel_size=5,
        alpha_d=6,
        save_path='order_stat_mixed_comparison.png'
    )

    # 测试6：d值效果演示
    print("\n测试6: 修正阿尔法均值滤波器d值效果")
    filters.demonstrate_alpha_d_effect(
        sp_noisy,
        kernel_size=5,
        save_path='order_stat_alpha_d_effect.png'
    )

    print("\n✅ 演示完成！")
    print(f"\n生成文件:")
    print(f"  - order_stat_sp_noisy.png (椒盐噪声图像)")
    print(f"  - order_stat_sp_comparison.png (椒盐噪声滤波对比)")
    print(f"  - order_stat_salt_noisy.png (盐噪声图像)")
    print(f"  - order_stat_salt_comparison.png (盐噪声滤波对比)")
    print(f"  - order_stat_pepper_noisy.png (椒噪声图像)")
    print(f"  - order_stat_pepper_comparison.png (椒噪声滤波对比)")
    print(f"  - order_stat_gaussian_noisy.png (高斯噪声图像)")
    print(f"  - order_stat_gaussian_comparison.png (高斯噪声滤波对比)")
    print(f"  - order_stat_mixed_noisy.png (混合噪声图像)")
    print(f"  - order_stat_mixed_comparison.png (混合噪声滤波对比)")
    print(f"  - order_stat_alpha_d_effect.png (d值效果对比)")


if __name__ == '__main__':
    main()
