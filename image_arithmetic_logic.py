"""
图像算术和逻辑操作实现
功能：
1. 算术操作：加法、减法、乘法、除法
2. 逻辑操作：与、或、非、异或
3. 支持单通道和多通道图像
4. 自动处理溢出和边界情况
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ImageArithmeticLogic:
    """图像算术和逻辑操作类"""

    def __init__(self):
        """初始化"""
        self.image1 = None
        self.image2 = None

    def load_images(self, path1: str, path2: Optional[str] = None) -> bool:
        """
        加载图像

        Args:
            path1: 第一张图像路径
            path2: 第二张图像路径（可选，用于双目操作）

        Returns:
            bool: 是否成功加载
        """
        self.image1 = cv2.imread(path1)
        if self.image1 is None:
            print(f"无法加载图像: {path1}")
            return False

        if path2 is not None:
            self.image2 = cv2.imread(path2)
            if self.image2 is None:
                print(f"无法加载图像: {path2}")
                return False

            # 确保两张图像尺寸一致
            if self.image1.shape != self.image2.shape:
                print("两张图像尺寸不一致，调整第二张图像尺寸")
                self.image2 = cv2.resize(self.image2,
                                        (self.image1.shape[1], self.image1.shape[0]))

        return True

    # ==================== 算术操作 ====================

    def add(self, image1: np.ndarray, image2: np.ndarray,
            weight1: float = 1.0, weight2: float = 1.0) -> np.ndarray:
        """
        图像加法：result = weight1 * image1 + weight2 * image2

        Args:
            image1: 第一张图像
            image2: 第二张图像
            weight1: 第一张图像权重
            weight2: 第二张图像权重

        Returns:
            np.ndarray: 相加后的图像（自动处理溢出）
        """
        # 使用cv2.addWeighted自动处理溢出
        result = cv2.addWeighted(image1, weight1, image2, weight2, 0)
        return result

    def subtract(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        图像减法：result = image1 - image2

        Args:
            image1: 被减图像
            image2: 减数图像

        Returns:
            np.ndarray: 相减后的图像（自动处理下溢）
        """
        # 使用cv2.subtract自动处理下溢
        result = cv2.subtract(image1, image2)
        return result

    def multiply(self, image1: np.ndarray, image2: np.ndarray,
                 scale: float = 1.0) -> np.ndarray:
        """
        图像乘法：result = (image1 * image2) * scale

        Args:
            image1: 第一张图像
            image2: 第二张图像
            scale: 缩放因子（用于归一化）

        Returns:
            np.ndarray: 相乘后的图像
        """
        # 转换为float避免溢出
        result = image1.astype(np.float32) * image2.astype(np.float32)
        result = result * scale

        # 归一化到0-255
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def divide(self, image1: np.ndarray, image2: np.ndarray,
               scale: float = 255.0) -> np.ndarray:
        """
        图像除法：result = (image1 / image2) * scale

        Args:
            image1: 被除图像
            image2: 除数图像
            scale: 缩放因子（用于可视化）

        Returns:
            np.ndarray: 相除后的图像（自动处理除零）
        """
        # 转换为float避免整数除法
        img1_float = image1.astype(np.float32)
        img2_float = image2.astype(np.float32)

        # 避免除零：将0替换为1
        img2_float[img2_float == 0] = 1

        result = (img1_float / img2_float) * scale
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    # ==================== 逻辑操作 ====================

    def bitwise_and(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        按位与操作：result = image1 AND image2

        Args:
            image1: 第一张图像
            image2: 第二张图像

        Returns:
            np.ndarray: 按位与结果
        """
        return cv2.bitwise_and(image1, image2)

    def bitwise_or(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        按位或操作：result = image1 OR image2

        Args:
            image1: 第一张图像
            image2: 第二张图像

        Returns:
            np.ndarray: 按位或结果
        """
        return cv2.bitwise_or(image1, image2)

    def bitwise_not(self, image: np.ndarray) -> np.ndarray:
        """
        按位非操作：result = NOT image

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 按位非结果
        """
        return cv2.bitwise_not(image)

    def bitwise_xor(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        按位异或操作：result = image1 XOR image2

        Args:
            image1: 第一张图像
            image2: 第二张图像

        Returns:
            np.ndarray: 按位异或结果
        """
        return cv2.bitwise_xor(image1, image2)

    # ==================== 可视化 ====================

    def visualize_arithmetic_operations(self, image1: np.ndarray,
                                       image2: np.ndarray,
                                       save_path: Optional[str] = None):
        """
        可视化所有算术操作

        Args:
            image1: 第一张图像
            image2: 第二张图像
            save_path: 保存路径（可选）
        """
        # 执行所有算术操作
        add_result = self.add(image1, image2, 0.5, 0.5)
        subtract_result = self.subtract(image1, image2)
        multiply_result = self.multiply(image1, image2, 1.0/255.0)
        divide_result = self.divide(image1, image2, 255.0)

        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('图像算术操作', fontsize=16, fontweight='bold')

        # 转换BGR到RGB用于显示
        images = [image1, image2, add_result, subtract_result,
                 multiply_result, divide_result]
        titles = ['原图像1', '原图像2', '加法 (0.5*I1 + 0.5*I2)',
                 '减法 (I1 - I2)', '乘法 (I1 * I2 / 255)', '除法 (I1 / I2 * 255)']

        for idx, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            ax.imshow(img_rgb, cmap='gray' if len(img.shape) == 2 else None)
            ax.set_title(title, fontsize=12)
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"算术操作结果已保存到: {save_path}")

        plt.close()

    def visualize_logic_operations(self, image1: np.ndarray,
                                   image2: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        可视化所有逻辑操作

        Args:
            image1: 第一张图像
            image2: 第二张图像
            save_path: 保存路径（可选）
        """
        # 执行所有逻辑操作
        and_result = self.bitwise_and(image1, image2)
        or_result = self.bitwise_or(image1, image2)
        not_result1 = self.bitwise_not(image1)
        xor_result = self.bitwise_xor(image1, image2)

        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('图像逻辑操作', fontsize=16, fontweight='bold')

        # 转换BGR到RGB用于显示
        images = [image1, image2, and_result, or_result, not_result1, xor_result]
        titles = ['原图像1', '原图像2', '按位与 (I1 AND I2)',
                 '按位或 (I1 OR I2)', '按位非 (NOT I1)', '按位异或 (I1 XOR I2)']

        for idx, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            ax.imshow(img_rgb, cmap='gray' if len(img.shape) == 2 else None)
            ax.set_title(title, fontsize=12)
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"逻辑操作结果已保存到: {save_path}")

        plt.close()


def main():
    """主函数：演示图像算术和逻辑操作"""

    # 创建操作对象
    processor = ImageArithmeticLogic()

    # 创建两张测试图像
    print("创建测试图像...")

    # 图像1：渐变图
    image1 = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(300):
        image1[i, :] = [i * 255 // 300, 128, 255 - i * 255 // 300]

    # 图像2：圆形图案
    image2 = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(image2, (150, 150), 100, (255, 255, 255), -1)
    cv2.circle(image2, (150, 150), 50, (128, 128, 128), -1)

    # 保存测试图像
    cv2.imwrite('test_image1.png', image1)
    cv2.imwrite('test_image2.png', image2)
    print("测试图像已创建")

    # 可视化算术操作
    print("\n执行算术操作...")
    processor.visualize_arithmetic_operations(
        image1, image2,
        save_path='arithmetic_operations.png'
    )

    # 可视化逻辑操作
    print("\n执行逻辑操作...")
    processor.visualize_logic_operations(
        image1, image2,
        save_path='logic_operations.png'
    )

    print("\n所有操作完成！")


if __name__ == "__main__":
    main()
