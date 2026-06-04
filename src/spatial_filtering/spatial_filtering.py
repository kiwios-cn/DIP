"""
空间域滤波实现
功能：
1. 均值滤波：标准均值、加权均值
2. 中值滤波：标准中值、自适应中值
3. 拉普拉斯滤波：标准拉普拉斯、锐化、LoG
4. 支持自定义卷积核
5. 多种边界处理方式
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from scipy import ndimage
from scipy.ndimage import median_filter as scipy_median

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SpatialFiltering:
    """空间域滤波类"""

    def __init__(self):
        """初始化"""
        self.image = None
        self.gray_image = None

    def load_image(self, path: str) -> bool:
        """
        加载图像

        Args:
            path: 图像路径

        Returns:
            bool: 是否成功加载
        """
        self.image = cv2.imread(path)
        if self.image is None:
            print(f"无法加载图像: {path}")
            return False

        # 转换为灰度图
        if len(self.image.shape) == 3:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_image = self.image.copy()

        return True

    # ==================== 均值滤波 ====================

    def mean_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        标准均值滤波（算术平均）

        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数）

        Returns:
            np.ndarray: 滤波后的图像
        """
        # 使用cv2.blur实现均值滤波
        result = cv2.blur(image, (kernel_size, kernel_size))
        return result

    def weighted_mean_filter(self, image: np.ndarray,
                            kernel_size: int = 3) -> np.ndarray:
        """
        加权均值滤波（高斯加权）

        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数）

        Returns:
            np.ndarray: 滤波后的图像
        """
        # 使用高斯滤波实现加权均值
        # sigma = 0 表示根据kernel_size自动计算
        result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return result

    def custom_mean_filter(self, image: np.ndarray,
                          kernel: np.ndarray) -> np.ndarray:
        """
        自定义卷积核的均值滤波

        Args:
            image: 输入图像
            kernel: 自定义卷积核

        Returns:
            np.ndarray: 滤波后的图像
        """
        # 归一化卷积核
        kernel = kernel / np.sum(kernel)
        result = cv2.filter2D(image, -1, kernel)
        return result

    # ==================== 中值滤波 ====================

    def median_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        标准中值滤波

        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数）

        Returns:
            np.ndarray: 滤波后的图像
        """
        result = cv2.medianBlur(image, kernel_size)
        return result

    def adaptive_median_filter(self, image: np.ndarray,
                              max_kernel_size: int = 7) -> np.ndarray:
        """
        自适应中值滤波
        根据局部统计特性自适应调整窗口大小

        Args:
            image: 输入图像
            max_kernel_size: 最大窗口大小

        Returns:
            np.ndarray: 滤波后的图像
        """
        h, w = image.shape
        result = image.copy()
        pad = max_kernel_size // 2

        # 边界填充
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                    cv2.BORDER_REFLECT)

        for i in range(h):
            for j in range(w):
                # 从小窗口开始
                for k_size in range(3, max_kernel_size + 1, 2):
                    half = k_size // 2
                    window = padded[i:i+k_size, j:j+k_size]

                    z_min = np.min(window)
                    z_max = np.max(window)
                    z_med = np.median(window)
                    z_xy = padded[i+pad, j+pad]

                    # 判断中值是否为噪声
                    if z_min < z_med < z_max:
                        # 中值不是噪声，判断当前像素
                        if z_min < z_xy < z_max:
                            result[i, j] = z_xy
                        else:
                            result[i, j] = z_med
                        break
                    # 否则增大窗口继续
                else:
                    # 达到最大窗口，使用中值
                    result[i, j] = z_med

        return result

    # ==================== 拉普拉斯滤波 ====================

    def laplacian_filter(self, image: np.ndarray,
                        kernel_type: str = '4') -> np.ndarray:
        """
        拉普拉斯滤波

        Args:
            image: 输入图像
            kernel_type: 卷积核类型
                '4' - 4邻域拉普拉斯
                '8' - 8邻域拉普拉斯
                'enhanced' - 增强型拉普拉斯

        Returns:
            np.ndarray: 拉普拉斯响应（可能有负值）
        """
        if kernel_type == '4':
            # 4邻域拉普拉斯算子
            kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float32)
        elif kernel_type == '8':
            # 8邻域拉普拉斯算子
            kernel = np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]], dtype=np.float32)
        elif kernel_type == 'enhanced':
            # 增强型拉普拉斯算子（对角线加权）
            kernel = np.array([[1, 4, 1],
                             [4, -20, 4],
                             [1, 4, 1]], dtype=np.float32)
        else:
            raise ValueError(f"未知的kernel_type: {kernel_type}")

        # 应用卷积
        result = cv2.filter2D(image, cv2.CV_32F, kernel)
        return result

    def laplacian_sharpen(self, image: np.ndarray,
                         kernel_type: str = '4',
                         alpha: float = 1.0) -> np.ndarray:
        """
        拉普拉斯锐化
        锐化图像 = 原图像 - alpha * 拉普拉斯响应

        Args:
            image: 输入图像
            kernel_type: 拉普拉斯核类型
            alpha: 锐化强度

        Returns:
            np.ndarray: 锐化后的图像
        """
        # 计算拉普拉斯响应
        laplacian = self.laplacian_filter(image, kernel_type)

        # 锐化：原图 - alpha * 拉普拉斯
        sharpened = image.astype(np.float32) - alpha * laplacian

        # 截断到有效范围
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened

    def log_filter(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        LoG滤波器（Laplacian of Gaussian）
        先高斯平滑，再拉普拉斯

        Args:
            image: 输入图像
            sigma: 高斯标准差

        Returns:
            np.ndarray: LoG响应
        """
        # 先高斯平滑
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)

        # 再拉普拉斯
        log_response = self.laplacian_filter(blurred, '8')
        return log_response

    # ==================== 辅助函数 ====================

    def add_noise(self, image: np.ndarray, noise_type: str = 'gaussian',
                 **kwargs) -> np.ndarray:
        """
        添加噪声

        Args:
            image: 输入图像
            noise_type: 噪声类型
                'gaussian' - 高斯噪声
                'salt_pepper' - 椒盐噪声
            **kwargs: 噪声参数

        Returns:
            np.ndarray: 添加噪声后的图像
        """
        noisy = image.copy().astype(np.float32)

        if noise_type == 'gaussian':
            mean = kwargs.get('mean', 0)
            sigma = kwargs.get('sigma', 25)
            noise = np.random.normal(mean, sigma, image.shape)
            noisy = noisy + noise

        elif noise_type == 'salt_pepper':
            prob = kwargs.get('prob', 0.05)
            # 椒噪声（黑点）
            salt = np.random.random(image.shape) < prob / 2
            noisy[salt] = 255
            # 盐噪声（白点）
            pepper = np.random.random(image.shape) < prob / 2
            noisy[pepper] = 0

        # 截断到有效范围
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    # ==================== 可视化 ====================

    def visualize_all_filters(self, image: np.ndarray,
                             save_path: Optional[str] = None):
        """
        可视化所有滤波器效果

        Args:
            image: 输入图像
            save_path: 保存路径（可选）
        """
        # 添加噪声
        gaussian_noisy = self.add_noise(image, 'gaussian', sigma=25)
        sp_noisy = self.add_noise(image, 'salt_pepper', prob=0.05)

        # 应用各种滤波器
        mean_3x3 = self.mean_filter(gaussian_noisy, 3)
        mean_5x5 = self.mean_filter(gaussian_noisy, 5)
        weighted_mean = self.weighted_mean_filter(gaussian_noisy, 5)

        median_3x3 = self.median_filter(sp_noisy, 3)
        median_5x5 = self.median_filter(sp_noisy, 5)

        laplacian_4 = self.laplacian_filter(image, '4')
        laplacian_8 = self.laplacian_filter(image, '8')
        sharpened = self.laplacian_sharpen(image, '4', alpha=1.0)
        log_result = self.log_filter(image, sigma=1.5)

        # 创建子图
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle('空间域滤波效果对比', fontsize=16, fontweight='bold')

        images = [
            (image, '原始图像'),
            (gaussian_noisy, '高斯噪声'),
            (sp_noisy, '椒盐噪声'),
            (mean_3x3, '均值滤波 3×3'),

            (mean_5x5, '均值滤波 5×5'),
            (weighted_mean, '加权均值 5×5'),
            (median_3x3, '中值滤波 3×3'),
            (median_5x5, '中值滤波 5×5'),

            (laplacian_4, '拉普拉斯 4邻域'),
            (laplacian_8, '拉普拉斯 8邻域'),
            (sharpened, '拉普拉斯锐化'),
            (log_result, 'LoG滤波'),

            (np.abs(laplacian_4).astype(np.uint8), '拉普拉斯绝对值'),
            (np.abs(log_result).astype(np.uint8), 'LoG绝对值'),
            (image, ''),
            (image, '')
        ]

        for idx, (ax, (img, title)) in enumerate(zip(axes.flat, images)):
            if idx < 14:
                ax.imshow(img, cmap='gray')
                ax.set_title(title, fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"滤波结果已保存到: {save_path}")

        plt.close()


def main():
    """主函数：演示空间域滤波"""

    # 创建滤波器对象
    filter_obj = SpatialFiltering()

    # 创建测试图像
    print("创建测试图像...")
    test_image = np.zeros((300, 300), dtype=np.uint8)

    # 添加一些几何图形
    cv2.circle(test_image, (150, 150), 80, 200, -1)
    cv2.rectangle(test_image, (50, 50), (100, 100), 150, -1)
    cv2.rectangle(test_image, (200, 200), (250, 250), 180, -1)

    # 保存测试图像
    cv2.imwrite('test_spatial_filtering.png', test_image)
    print("测试图像已创建")

    # 可视化所有滤波器
    print("\n执行空间域滤波...")
    filter_obj.visualize_all_filters(
        test_image,
        save_path='spatial_filtering_results.png'
    )

    print("\n所有滤波操作完成！")


if __name__ == "__main__":
    main()
