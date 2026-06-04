"""
频域陷波滤波器实现
功能：
1. 理想陷波滤波器：在指定频率处完全阻止或通过
2. 巴特沃斯陷波滤波器：平滑过渡的陷波特性
3. 高斯陷波滤波器：高斯形状的陷波特性
4. 陷波抑制/通过模式：去除或保留特定频率
5. 周期噪声去除应用
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FrequencyNotchFilter:
    """频域陷波滤波器类"""

    def __init__(self):
        """初始化"""
        self.image = None
        self.gray_image = None
        self.fft_image = None
        self.fft_shifted = None

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

    # ==================== FFT变换 ====================

    def fft_transform(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        FFT变换并中心化

        Args:
            image: 输入图像

        Returns:
            Tuple[np.ndarray, np.ndarray]: (FFT结果, 中心化FFT结果)
        """
        # FFT变换
        fft = np.fft.fft2(image)
        # 中心化（将零频率移到中心）
        fft_shifted = np.fft.fftshift(fft)

        return fft, fft_shifted

    def ifft_transform(self, fft_shifted: np.ndarray) -> np.ndarray:
        """
        逆FFT变换

        Args:
            fft_shifted: 中心化的FFT结果

        Returns:
            np.ndarray: 恢复的图像
        """
        # 逆中心化
        fft = np.fft.ifftshift(fft_shifted)
        # 逆FFT
        image = np.fft.ifft2(fft)
        # 取实部
        image = np.abs(image)

        return image

    # ==================== 陷波滤波器核心 ====================

    def create_notch_mask(self, shape: Tuple[int, int],
                         notch_positions: List[Tuple[int, int]],
                         radius: float,
                         filter_type: str = 'ideal',
                         order: int = 2) -> np.ndarray:
        """
        创建陷波掩码（自动创建对称陷波点对）

        Args:
            shape: 图像形状 (H, W)
            notch_positions: 陷波位置列表 [(u1, v1), (u2, v2), ...]
                           只需指定一侧，自动创建关于中心对称的点对
            radius: 陷波半径
            filter_type: 滤波器类型
                'ideal' - 理想陷波
                'butterworth' - 巴特沃斯陷波
                'gaussian' - 高斯陷波
            order: 巴特沃斯滤波器的阶数

        Returns:
            np.ndarray: 陷波掩码（1=通过，0=阻止）
        """
        H, W = shape
        # 创建频率坐标（以中心为原点）
        u = np.arange(H) - H // 2
        v = np.arange(W) - W // 2
        U, V = np.meshgrid(v, u)

        # 初始化掩码（全1表示全部通过）
        mask = np.ones((H, W), dtype=np.float32)

        # 中心点
        center_u, center_v = H // 2, W // 2

        # 对每个陷波位置，创建对称点对
        for (u_notch, v_notch) in notch_positions:
            # 计算陷波点相对中心的偏移
            du = u_notch - center_u
            dv = v_notch - center_v

            # 创建两个对称的陷波点
            notch_points = [
                (u_notch, v_notch),           # 原始点
                (center_u - du, center_v - dv)  # 对称点
            ]

            for (un, vn) in notch_points:
                # 计算到陷波点的距离
                D = np.sqrt((U - (vn - center_v))**2 + (V - (un - center_u))**2)

                if filter_type == 'ideal':
                    # 理想陷波：圆内为0，圆外为1
                    notch = np.where(D <= radius, 0, 1)

                elif filter_type == 'butterworth':
                    # 巴特沃斯陷波：H = 1 / (1 + (D0/D)^(2n))
                    # D0是陷波半径，D是到陷波点的距离
                    # 当D很小时，H接近0（阻止）
                    # 使用倒数形式避免除零
                    with np.errstate(divide='ignore', invalid='ignore'):
                        notch = 1.0 / (1.0 + (radius / (D + 1e-8))**(2 * order))
                    notch = np.nan_to_num(notch)

                elif filter_type == 'gaussian':
                    # 高斯陷波：H = 1 - exp(-(D^2)/(2*D0^2))
                    notch = 1.0 - np.exp(-(D**2) / (2 * radius**2))

                else:
                    raise ValueError(f"未知的滤波器类型: {filter_type}")

                # 与现有掩码相乘（多个陷波点的叠加效果）
                mask *= notch

        return mask

    def ideal_notch_reject(self, image: np.ndarray,
                          notch_positions: List[Tuple[int, int]],
                          radius: float) -> np.ndarray:
        """
        理想陷波抑制滤波器

        Args:
            image: 输入图像
            notch_positions: 陷波位置列表
            radius: 陷波半径

        Returns:
            np.ndarray: 滤波后的图像
        """
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 创建陷波掩码
        mask = self.create_notch_mask(image.shape, notch_positions, radius, 'ideal')

        # 应用滤波器
        filtered_fft = fft_shifted * mask

        # 逆变换
        result = self.ifft_transform(filtered_fft)

        return result.astype(np.uint8)

    def butterworth_notch_reject(self, image: np.ndarray,
                                notch_positions: List[Tuple[int, int]],
                                radius: float,
                                order: int = 2) -> np.ndarray:
        """
        巴特沃斯陷波抑制滤波器

        Args:
            image: 输入图像
            notch_positions: 陷波位置列表
            radius: 陷波半径
            order: 滤波器阶数

        Returns:
            np.ndarray: 滤波后的图像
        """
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 创建陷波掩码
        mask = self.create_notch_mask(image.shape, notch_positions, radius,
                                     'butterworth', order)

        # 应用滤波器
        filtered_fft = fft_shifted * mask

        # 逆变换
        result = self.ifft_transform(filtered_fft)

        return result.astype(np.uint8)

    def gaussian_notch_reject(self, image: np.ndarray,
                             notch_positions: List[Tuple[int, int]],
                             radius: float) -> np.ndarray:
        """
        高斯陷波抑制滤波器

        Args:
            image: 输入图像
            notch_positions: 陷波位置列表
            radius: 陷波半径

        Returns:
            np.ndarray: 滤波后的图像
        """
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 创建陷波掩码
        mask = self.create_notch_mask(image.shape, notch_positions, radius, 'gaussian')

        # 应用滤波器
        filtered_fft = fft_shifted * mask

        # 逆变换
        result = self.ifft_transform(filtered_fft)

        return result.astype(np.uint8)

    def ideal_notch_pass(self, image: np.ndarray,
                        notch_positions: List[Tuple[int, int]],
                        radius: float) -> np.ndarray:
        """
        理想陷波通过滤波器（只保留指定频率）

        Args:
            image: 输入图像
            notch_positions: 陷波位置列表
            radius: 陷波半径

        Returns:
            np.ndarray: 滤波后的图像
        """
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 创建陷波抑制掩码
        reject_mask = self.create_notch_mask(image.shape, notch_positions, radius, 'ideal')

        # 陷波通过掩码 = 1 - 陷波抑制掩码
        pass_mask = 1.0 - reject_mask

        # 应用滤波器
        filtered_fft = fft_shifted * pass_mask

        # 逆变换
        result = self.ifft_transform(filtered_fft)

        return result.astype(np.uint8)

    # ==================== 辅助函数 ====================

    def add_periodic_noise(self, image: np.ndarray,
                          frequencies: List[Tuple[float, float]],
                          amplitudes: List[float]) -> np.ndarray:
        """
        添加周期性噪声

        Args:
            image: 输入图像
            frequencies: 频率列表 [(freq_u, freq_v), ...]
            amplitudes: 幅度列表

        Returns:
            np.ndarray: 添加噪声后的图像
        """
        H, W = image.shape
        noisy = image.astype(np.float32)

        for (freq_u, freq_v), amp in zip(frequencies, amplitudes):
            # 创建正弦噪声
            y = np.arange(H)
            x = np.arange(W)
            X, Y = np.meshgrid(x, y)

            noise = amp * np.sin(2 * np.pi * (freq_u * Y / H + freq_v * X / W))
            noisy += noise

        # 截断到有效范围
        noisy = np.clip(noisy, 0, 255)
        return noisy.astype(np.uint8)

    def detect_peaks_in_spectrum(self, fft_shifted: np.ndarray,
                                 threshold_percentile: float = 99.9,
                                 min_distance: int = 10) -> List[Tuple[int, int]]:
        """
        在频谱中检测峰值（噪声频率位置）

        Args:
            fft_shifted: 中心化的FFT结果
            threshold_percentile: 峰值阈值百分位
            min_distance: 峰值之间的最小距离

        Returns:
            List[Tuple[int, int]]: 峰值位置列表（排除DC分量）
        """
        # 计算幅度谱
        magnitude = np.abs(fft_shifted)

        # 计算阈值
        threshold = np.percentile(magnitude, threshold_percentile)

        # 找到超过阈值的点
        peaks = magnitude > threshold

        # 排除中心点（DC分量）
        H, W = magnitude.shape
        center_u, center_v = H // 2, W // 2
        peaks[center_u, center_v] = False

        # 排除中心附近的点
        for du in range(-2, 3):
            for dv in range(-2, 3):
                if 0 <= center_u + du < H and 0 <= center_v + dv < W:
                    peaks[center_u + du, center_v + dv] = False

        # 获取峰值坐标
        peak_coords = np.argwhere(peaks)

        # 过滤距离太近的峰值
        filtered_peaks = []
        for coord in peak_coords:
            u, v = coord
            # 检查与已选峰值的距离
            too_close = False
            for (uf, vf) in filtered_peaks:
                if np.sqrt((u - uf)**2 + (v - vf)**2) < min_distance:
                    too_close = True
                    break
            if not too_close:
                filtered_peaks.append((u, v))

        return filtered_peaks

    # ==================== 可视化 ====================

    def visualize_spectrum(self, image: np.ndarray,
                          title: str = "频谱",
                          save_path: Optional[str] = None):
        """
        可视化频谱

        Args:
            image: 输入图像
            title: 标题
            save_path: 保存路径（可选）
        """
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 计算幅度谱（对数尺度）
        magnitude = np.abs(fft_shifted)
        magnitude_log = np.log(1 + magnitude)

        # 显示
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('原始图像')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(magnitude_log, cmap='gray')
        plt.title(title)
        plt.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"频谱图已保存到: {save_path}")

        plt.close()

    def visualize_notch_filtering(self, image: np.ndarray,
                                  notch_positions: List[Tuple[int, int]],
                                  radius: float,
                                  save_path: Optional[str] = None):
        """
        可视化陷波滤波效果

        Args:
            image: 输入图像
            notch_positions: 陷波位置列表
            radius: 陷波半径
            save_path: 保存路径（可选）
        """
        # 应用三种陷波滤波器
        ideal_result = self.ideal_notch_reject(image, notch_positions, radius)
        butterworth_result = self.butterworth_notch_reject(image, notch_positions, radius, order=2)
        gaussian_result = self.gaussian_notch_reject(image, notch_positions, radius)

        # 获取频谱
        _, fft_original = self.fft_transform(image)
        _, fft_ideal = self.fft_transform(ideal_result)
        _, fft_butterworth = self.fft_transform(butterworth_result)
        _, fft_gaussian = self.fft_transform(gaussian_result)

        # 创建陷波掩码用于显示
        mask_ideal = self.create_notch_mask(image.shape, notch_positions, radius, 'ideal')
        mask_butterworth = self.create_notch_mask(image.shape, notch_positions, radius, 'butterworth')
        mask_gaussian = self.create_notch_mask(image.shape, notch_positions, radius, 'gaussian')

        # 显示
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('陷波滤波器效果对比', fontsize=16, fontweight='bold')

        # 原始图像和频谱
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('原始图像（含噪声）')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(np.log(1 + np.abs(fft_original)), cmap='gray')
        axes[0, 1].set_title('原始频谱')
        axes[0, 1].axis('off')

        # 理想陷波
        axes[0, 2].imshow(mask_ideal, cmap='gray')
        axes[0, 2].set_title('理想陷波掩码')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(ideal_result, cmap='gray')
        axes[0, 3].set_title('理想陷波结果')
        axes[0, 3].axis('off')

        # 巴特沃斯陷波
        axes[1, 0].imshow(np.log(1 + np.abs(fft_ideal)), cmap='gray')
        axes[1, 0].set_title('理想陷波频谱')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(mask_butterworth, cmap='gray')
        axes[1, 1].set_title('巴特沃斯陷波掩码')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(butterworth_result, cmap='gray')
        axes[1, 2].set_title('巴特沃斯陷波结果')
        axes[1, 2].axis('off')

        axes[1, 3].imshow(np.log(1 + np.abs(fft_butterworth)), cmap='gray')
        axes[1, 3].set_title('巴特沃斯陷波频谱')
        axes[1, 3].axis('off')

        # 高斯陷波
        axes[2, 0].imshow(mask_gaussian, cmap='gray')
        axes[2, 0].set_title('高斯陷波掩码')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(gaussian_result, cmap='gray')
        axes[2, 1].set_title('高斯陷波结果')
        axes[2, 1].axis('off')

        axes[2, 2].imshow(np.log(1 + np.abs(fft_gaussian)), cmap='gray')
        axes[2, 2].set_title('高斯陷波频谱')
        axes[2, 2].axis('off')

        # 标记陷波位置
        H, W = image.shape
        center_u, center_v = H // 2, W // 2
        ax = axes[2, 3]
        ax.imshow(np.log(1 + np.abs(fft_original)), cmap='gray')
        for (u, v) in notch_positions:
            # 原始点
            ax.plot(v - center_v + W//2, u - center_u + H//2, 'ro', markersize=8)
            # 对称点
            ax.plot(center_v - (v - center_v) + W//2, center_u - (u - center_u) + H//2,
                   'ro', markersize=8)
        ax.set_title('陷波位置标记')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"陷波滤波对比图已保存到: {save_path}")

        plt.close()


def main():
    """主函数：演示陷波滤波"""

    # 创建滤波器对象
    filter_obj = FrequencyNotchFilter()

    # 创建测试图像
    print("创建测试图像...")
    H, W = 256, 256
    test_image = np.zeros((H, W), dtype=np.uint8)

    # 添加一些内容
    cv2.circle(test_image, (128, 128), 60, 150, -1)
    cv2.rectangle(test_image, (50, 50), (100, 100), 200, -1)

    # 添加周期性噪声（正弦噪声）
    print("添加周期性噪声...")
    noisy_image = filter_obj.add_periodic_noise(
        test_image,
        frequencies=[(0, 30), (30, 0), (20, 20)],  # 三个频率的噪声
        amplitudes=[30, 30, 20]
    )

    # 保存噪声图像
    cv2.imwrite('notch_filter_noisy.png', noisy_image)
    print("噪声图像已保存")

    # 可视化频谱
    print("\n可视化频谱...")
    filter_obj.visualize_spectrum(
        noisy_image,
        title="带周期噪声的频谱",
        save_path='notch_filter_spectrum.png'
    )

    # 自动检测噪声峰值
    print("\n检测频谱峰值...")
    _, fft_shifted = filter_obj.fft_transform(noisy_image)
    peaks = filter_obj.detect_peaks_in_spectrum(fft_shifted, threshold_percentile=99.5)
    print(f"检测到 {len(peaks)} 个峰值: {peaks[:10]}")  # 只显示前10个

    # 手动指定陷波位置（基于已知噪声频率）
    # 噪声频率转换为FFT坐标
    notch_positions = [
        (128, 128 + 30),  # (0, 30) 频率
        (128 + 30, 128),  # (30, 0) 频率
        (128 + 20, 128 + 20)  # (20, 20) 频率
    ]

    print(f"\n使用陷波位置: {notch_positions}")
    print("应用陷波滤波器...")

    # 可视化陷波滤波效果
    filter_obj.visualize_notch_filtering(
        noisy_image,
        notch_positions=notch_positions,
        radius=10,
        save_path='notch_filter_results.png'
    )

    print("\n陷波滤波演示完成！")


if __name__ == "__main__":
    main()
