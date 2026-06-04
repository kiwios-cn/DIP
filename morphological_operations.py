"""
形态学处理模块

实现形态学图像处理操作：
1. 腐蚀 (Erosion)
2. 膨胀 (Dilation)
3. 开运算 (Opening)
4. 闭运算 (Closing)
5. 形态学梯度 (Morphological Gradient)
6. 顶帽变换 (Top Hat)
7. 黑帽变换 (Black Hat)
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Literal
import matplotlib.pyplot as plt


class MorphologicalOperations:
    """形态学操作类"""

    def __init__(self):
        self.image = None

    # ==================== 基本形态学操作 ====================

    def erosion(self, image: np.ndarray,
                kernel_size: int = 3,
                kernel_shape: str = 'rect',
                iterations: int = 1) -> np.ndarray:
        """
        腐蚀操作

        用结构元素扫描图像，只有当结构元素完全包含在前景内时，
        中心像素才保留为前景。

        公式: dst = erode(src) = min{src(x+x',y+y') | (x',y')∈kernel}

        Args:
            image: 输入图像（二值或灰度）
            kernel_size: 结构元素大小
            kernel_shape: 结构元素形状 'rect', 'ellipse', 'cross'
            iterations: 迭代次数

        Returns:
            np.ndarray: 腐蚀后的图像

        应用场景：
            - 去除小的白色噪声点
            - 分离连接的物体
            - 缩小前景区域
            - 提取骨架

        效果：
            - 缩小亮区域（白色区域）
            - 扩大暗区域（黑色区域）
            - 去除小的突出部分
        """
        kernel = self._get_kernel(kernel_size, kernel_shape)
        result = cv2.erode(image, kernel, iterations=iterations)
        return result

    def dilation(self, image: np.ndarray,
                 kernel_size: int = 3,
                 kernel_shape: str = 'rect',
                 iterations: int = 1) -> np.ndarray:
        """
        膨胀操作

        用结构元素扫描图像，只要结构元素与前景有任何交集，
        中心像素就设为前景。

        公式: dst = dilate(src) = max{src(x+x',y+y') | (x',y')∈kernel}

        Args:
            image: 输入图像（二值或灰度）
            kernel_size: 结构元素大小
            kernel_shape: 结构元素形状
            iterations: 迭代次数

        Returns:
            np.ndarray: 膨胀后的图像

        应用场景：
            - 填充小的黑色孔洞
            - 连接邻近的物体
            - 扩大前景区域
            - 增强物体轮廓

        效果：
            - 扩大亮区域（白色区域）
            - 缩小暗区域（黑色区域）
            - 填充小的凹陷
        """
        kernel = self._get_kernel(kernel_size, kernel_shape)
        result = cv2.dilate(image, kernel, iterations=iterations)
        return result

    # ==================== 复合形态学操作 ====================

    def opening(self, image: np.ndarray,
                kernel_size: int = 3,
                kernel_shape: str = 'rect') -> np.ndarray:
        """
        开运算

        先腐蚀后膨胀。

        公式: opening = dilate(erode(src))

        Args:
            image: 输入图像
            kernel_size: 结构元素大小
            kernel_shape: 结构元素形状

        Returns:
            np.ndarray: 开运算结果

        应用场景：
            - 去除小的白色噪声点
            - 平滑物体轮廓
            - 断开细的连接
            - 去除小的突出

        特点：
            - 先腐蚀（去除小物体）
            - 后膨胀（恢复大物体大小）
            - 比原图小或相等
            - 不会引入新的白色区域
        """
        kernel = self._get_kernel(kernel_size, kernel_shape)
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return result

    def closing(self, image: np.ndarray,
                kernel_size: int = 3,
                kernel_shape: str = 'rect') -> np.ndarray:
        """
        闭运算

        先膨胀后腐蚀。

        公式: closing = erode(dilate(src))

        Args:
            image: 输入图像
            kernel_size: 结构元素大小
            kernel_shape: 结构元素形状

        Returns:
            np.ndarray: 闭运算结果

        应用场景：
            - 填充小的黑色孔洞
            - 连接邻近的物体
            - 平滑物体轮廓
            - 填充小的凹陷

        特点：
            - 先膨胀（填充孔洞）
            - 后腐蚀（恢复大小）
            - 比原图大或相等
            - 不会引入新的黑色区域
        """
        kernel = self._get_kernel(kernel_size, kernel_shape)
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return result

    # ==================== 高级形态学操作 ====================

    def morphological_gradient(self, image: np.ndarray,
                               kernel_size: int = 3,
                               kernel_shape: str = 'rect') -> np.ndarray:
        """
        形态学梯度

        膨胀图像与腐蚀图像的差。

        公式: gradient = dilate(src) - erode(src)

        Args:
            image: 输入图像
            kernel_size: 结构元素大小
            kernel_shape: 结构元素形状

        Returns:
            np.ndarray: 形态学梯度

        应用场景：
            - 提取物体边缘
            - 轮廓检测
            - 物体分割

        特点：
            - 突出物体边界
            - 边缘粗细取决于结构元素大小
        """
        kernel = self._get_kernel(kernel_size, kernel_shape)
        result = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        return result

    def top_hat(self, image: np.ndarray,
                kernel_size: int = 9,
                kernel_shape: str = 'rect') -> np.ndarray:
        """
        顶帽变换（白帽）

        原图像与开运算结果的差。

        公式: tophat = src - opening(src)

        Args:
            image: 输入图像
            kernel_size: 结构元素大小（通常较大）
            kernel_shape: 结构元素形状

        Returns:
            np.ndarray: 顶帽变换结果

        应用场景：
            - 提取小的亮区域
            - 增强局部亮点
            - 背景校正

        特点：
            - 提取比结构元素小的亮区域
            - 去除大的亮区域
        """
        kernel = self._get_kernel(kernel_size, kernel_shape)
        result = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        return result

    def black_hat(self, image: np.ndarray,
                  kernel_size: int = 9,
                  kernel_shape: str = 'rect') -> np.ndarray:
        """
        黑帽变换

        闭运算结果与原图像的差。

        公式: blackhat = closing(src) - src

        Args:
            image: 输入图像
            kernel_size: 结构元素大小（通常较大）
            kernel_shape: 结构元素形状

        Returns:
            np.ndarray: 黑帽变换结果

        应用场景：
            - 提取小的暗区域
            - 增强局部暗点
            - 背景校正

        特点：
            - 提取比结构元素小的暗区域
            - 去除大的暗区域
        """
        kernel = self._get_kernel(kernel_size, kernel_shape)
        result = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        return result

    # ==================== 辅助方法 ====================

    def _get_kernel(self, size: int, shape: str) -> np.ndarray:
        """
        获取结构元素

        Args:
            size: 结构元素大小
            shape: 形状 'rect', 'ellipse', 'cross'

        Returns:
            np.ndarray: 结构元素
        """
        if shape == 'rect':
            return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        elif shape == 'ellipse':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        elif shape == 'cross':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
        else:
            raise ValueError(f"不支持的结构元素形状: {shape}")

    def load_image(self, path: str, grayscale: bool = True) -> bool:
        """加载图像"""
        if grayscale:
            self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            self.image = cv2.imread(path)
        return self.image is not None

    # ==================== 可视化 ====================

    def compare_basic_operations(self, image: np.ndarray,
                                 kernel_size: int = 5,
                                 save_path: Optional[str] = None):
        """比较基本形态学操作"""
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # 原图
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')

        # 腐蚀
        eroded = self.erosion(image, kernel_size)
        axes[1].imshow(eroded, cmap='gray')
        axes[1].set_title(f'腐蚀 (kernel={kernel_size}×{kernel_size})', fontsize=12)
        axes[1].axis('off')

        # 膨胀
        dilated = self.dilation(image, kernel_size)
        axes[2].imshow(dilated, cmap='gray')
        axes[2].set_title(f'膨胀 (kernel={kernel_size}×{kernel_size})', fontsize=12)
        axes[2].axis('off')

        # 开运算
        opened = self.opening(image, kernel_size)
        axes[3].imshow(opened, cmap='gray')
        axes[3].set_title(f'开运算 (去除小白点)', fontsize=12)
        axes[3].axis('off')

        # 闭运算
        closed = self.closing(image, kernel_size)
        axes[4].imshow(closed, cmap='gray')
        axes[4].set_title(f'闭运算 (填充小黑洞)', fontsize=12)
        axes[4].axis('off')

        # 形态学梯度
        gradient = self.morphological_gradient(image, kernel_size)
        axes[5].imshow(gradient, cmap='gray')
        axes[5].set_title(f'形态学梯度 (边缘)', fontsize=12)
        axes[5].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"基本操作对比图已保存到: {save_path}")

        plt.show()

    def compare_advanced_operations(self, image: np.ndarray,
                                   kernel_size: int = 9,
                                   save_path: Optional[str] = None):
        """比较高级形态学操作"""
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        # 原图
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')

        # 形态学梯度
        gradient = self.morphological_gradient(image, kernel_size=5)
        axes[1].imshow(gradient, cmap='gray')
        axes[1].set_title('形态学梯度 (边缘提取)', fontsize=12)
        axes[1].axis('off')

        # 顶帽变换
        tophat = self.top_hat(image, kernel_size)
        axes[2].imshow(tophat, cmap='gray')
        axes[2].set_title(f'顶帽变换 (提取小亮点)', fontsize=12)
        axes[2].axis('off')

        # 黑帽变换
        blackhat = self.black_hat(image, kernel_size)
        axes[3].imshow(blackhat, cmap='gray')
        axes[3].set_title(f'黑帽变换 (提取小暗点)', fontsize=12)
        axes[3].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"高级操作对比图已保存到: {save_path}")

        plt.show()

    def demonstrate_kernel_shapes(self, image: np.ndarray,
                                  save_path: Optional[str] = None):
        """演示不同结构元素形状的效果"""
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        # 原图
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')

        # 矩形结构元素
        rect_result = self.erosion(image, kernel_size=5, kernel_shape='rect')
        axes[1].imshow(rect_result, cmap='gray')
        axes[1].set_title('矩形结构元素', fontsize=12)
        axes[1].axis('off')

        # 椭圆结构元素
        ellipse_result = self.erosion(image, kernel_size=5, kernel_shape='ellipse')
        axes[2].imshow(ellipse_result, cmap='gray')
        axes[2].set_title('椭圆结构元素', fontsize=12)
        axes[2].axis('off')

        # 十字结构元素
        cross_result = self.erosion(image, kernel_size=5, kernel_shape='cross')
        axes[3].imshow(cross_result, cmap='gray')
        axes[3].set_title('十字结构元素', fontsize=12)
        axes[3].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结构元素对比图已保存到: {save_path}")

        plt.show()


def main():
    """演示程序"""
    print("形态学处理演示程序")
    print("=" * 50)

    morph = MorphologicalOperations()

    # 创建测试二值图像
    print("\n创建测试二值图像...")
    test_binary = np.zeros((256, 256), dtype=np.uint8)

    # 绘制一些形状
    cv2.rectangle(test_binary, (50, 50), (150, 150), 255, -1)
    cv2.circle(test_binary, (200, 100), 40, 255, -1)
    cv2.rectangle(test_binary, (100, 180), (200, 230), 255, -1)

    # 添加一些噪声点
    noise_points = np.random.randint(0, 256, (50, 2))
    for point in noise_points:
        cv2.circle(test_binary, tuple(point), 2, 255, -1)

    cv2.imwrite('morph_test_binary.png', test_binary)

    # 测试1: 基本形态学操作
    print("\n测试1: 基本形态学操作")
    morph.compare_basic_operations(test_binary, kernel_size=5,
                                   save_path='morph_basic_operations.png')

    # 测试2: 高级形态学操作
    print("\n测试2: 高级形态学操作")
    morph.compare_advanced_operations(test_binary, kernel_size=9,
                                     save_path='morph_advanced_operations.png')

    # 测试3: 不同结构元素形状
    print("\n测试3: 不同结构元素形状")
    morph.demonstrate_kernel_shapes(test_binary,
                                   save_path='morph_kernel_shapes.png')

    # 测试4: 去除噪声
    print("\n测试4: 噪声去除演示")
    # 添加椒盐噪声
    noisy = test_binary.copy()
    from noise_simulation import NoiseSimulation
    noise_sim = NoiseSimulation()
    noisy = noise_sim.add_salt_pepper_noise(test_binary, prob=0.05)

    # 使用开运算去除白噪声
    cleaned_opening = morph.opening(noisy, kernel_size=3)

    # 使用闭运算填充黑噪声
    cleaned_closing = morph.closing(noisy, kernel_size=3)

    # 先开后闭
    cleaned_both = morph.closing(morph.opening(noisy, 3), 3)

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('含噪声图像', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(cleaned_opening, cmap='gray')
    axes[1].set_title('开运算去噪', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(cleaned_closing, cmap='gray')
    axes[2].set_title('闭运算去噪', fontsize=12)
    axes[2].axis('off')

    axes[3].imshow(cleaned_both, cmap='gray')
    axes[3].set_title('开运算+闭运算', fontsize=12)
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('morph_noise_removal.png', dpi=150, bbox_inches='tight')
    print("噪声去除对比图已保存到: morph_noise_removal.png")
    plt.show()

    print("\n✅ 演示完成！")
    print(f"\n生成文件:")
    print(f"  - morph_test_binary.png (测试二值图像)")
    print(f"  - morph_basic_operations.png (基本操作对比)")
    print(f"  - morph_advanced_operations.png (高级操作对比)")
    print(f"  - morph_kernel_shapes.png (结构元素对比)")
    print(f"  - morph_noise_removal.png (噪声去除演示)")


if __name__ == '__main__':
    main()
