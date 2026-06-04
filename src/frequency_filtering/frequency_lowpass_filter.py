"""
频域低通/高通滤波器实现
功能：
1. 理想低通滤波器（ILPF）：完全截断高频
2. 巴特沃斯低通滤波器（BLPF）：平滑过渡
3. 高斯低通滤波器（GLPF）：高斯形状过渡
4. 对应的高通滤波器：IHPF、BHPF、GHPF
5. 带通/带阻滤波器
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FrequencyLowpassFilter:
    """频域低通/高通滤波器类"""

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

    # ==================== 低通滤波器 ====================

    def ideal_lowpass_filter(self, image: np.ndarray, cutoff: float) -> np.ndarray:
        """
        理想低通滤波器（ILPF）

        Args:
            image: 输入图像
            cutoff: 截止频率（距离中心的半径）

        Returns:
            np.ndarray: 滤波后的图像
        """
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 创建理想低通掩码
        H, W = image.shape
        mask = self._create_circular_mask(H, W, cutoff, filter_type='ideal', pass_type='low')

        # 应用滤波器
        filtered_fft = fft_shifted * mask

        # 逆变换
        result = self.ifft_transform(filtered_fft)

        return result.astype(np.uint8)

    def butterworth_lowpass_filter(self, image: np.ndarray,
                                   cutoff: float,
                                   order: int = 2) -> np.ndarray:
        """
        巴特沃斯低通滤波器（BLPF）

        Args:
            image: 输入图像
            cutoff: 截止频率
            order: 滤波器阶数

        Returns:
            np.ndarray: 滤波后的图像
        """
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 创建巴特沃斯低通掩码
        H, W = image.shape
        mask = self._create_circular_mask(H, W, cutoff, filter_type='butterworth',
                                         pass_type='low', order=order)

        # 应用滤波器
        filtered_fft = fft_shifted * mask

        # 逆变换
        result = self.ifft_transform(filtered_fft)

        return result.astype(np.uint8)

    def gaussian_lowpass_filter(self, image: np.ndarray, cutoff: float) -> np.ndarray:
        """
        高斯低通滤波器（GLPF）

        Args:
            image: 输入图像
            cutoff: 截止频率（标准差）

        Returns:
            np.ndarray: 滤波后的图像
        """
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 创建高斯低通掩码
        H, W = image.shape
        mask = self._create_circular_mask(H, W, cutoff, filter_type='gaussian', pass_type='low')

        # 应用滤波器
        filtered_fft = fft_shifted * mask

        # 逆变换
        result = self.ifft_transform(filtered_fft)

        return result.astype(np.uint8)

    # ==================== 高通滤波器 ====================

    def ideal_highpass_filter(self, image: np.ndarray, cutoff: float) -> np.ndarray:
        """
        理想高通滤波器（IHPF）

        Args:
            image: 输入图像
            cutoff: 截止频率

        Returns:
            np.ndarray: 滤波后的图像
        """
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 创建理想高通掩码
        H, W = image.shape
        mask = self._create_circular_mask(H, W, cutoff, filter_type='ideal', pass_type='high')

        # 应用滤波器
        filtered_fft = fft_shifted * mask

        # 逆变换
        result = self.ifft_transform(filtered_fft)

        return result.astype(np.uint8)

    def butterworth_highpass_filter(self, image: np.ndarray,
                                    cutoff: float,
                                    order: int = 2) -> np.ndarray:
        """
        巴特沃斯高通滤波器（BHPF）

        Args:
            image: 输入图像
            cutoff: 截止频率
            order: 滤波器阶数

        Returns:
            np.ndarray: 滤波后的图像
        """
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 创建巴特沃斯高通掩码
        H, W = image.shape
        mask = self._create_circular_mask(H, W, cutoff, filter_type='butterworth',
                                         pass_type='high', order=order)

        # 应用滤波器
        filtered_fft = fft_shifted * mask

        # 逆变换
        result = self.ifft_transform(filtered_fft)

        return result.astype(np.uint8)

    def gaussian_highpass_filter(self, image: np.ndarray, cutoff: float) -> np.ndarray:
        """
        高斯高通滤波器（GHPF）

        Args:
            image: 输入图像
            cutoff: 截止频率

        Returns:
            np.ndarray: 滤波后的图像
        """
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 创建高斯高通掩码
        H, W = image.shape
        mask = self._create_circular_mask(H, W, cutoff, filter_type='gaussian', pass_type='high')

        # 应用滤波器
        filtered_fft = fft_shifted * mask

        # 逆变换
        result = self.ifft_transform(filtered_fft)

        return result.astype(np.uint8)

    # ==================== 带通/带阻滤波器 ====================

    def bandpass_filter(self, image: np.ndarray,
                       cutoff_low: float,
                       cutoff_high: float,
                       filter_type: str = 'butterworth',
                       order: int = 2) -> np.ndarray:
        """
        带通滤波器（保留特定频率范围）

        Args:
            image: 输入图像
            cutoff_low: 低截止频率
            cutoff_high: 高截止频率
            filter_type: 滤波器类型
            order: 巴特沃斯阶数

        Returns:
            np.ndarray: 滤波后的图像
        """
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 创建带通掩码 = 高通 * 低通
        H, W = image.shape
        mask_high = self._create_circular_mask(H, W, cutoff_low, filter_type, 'high', order)
        mask_low = self._create_circular_mask(H, W, cutoff_high, filter_type, 'low', order)
        mask = mask_high * mask_low

        # 应用滤波器
        filtered_fft = fft_shifted * mask

        # 逆变换
        result = self.ifft_transform(filtered_fft)

        return result.astype(np.uint8)

    def bandreject_filter(self, image: np.ndarray,
                         cutoff_low: float,
                         cutoff_high: float,
                         filter_type: str = 'butterworth',
                         order: int = 2) -> np.ndarray:
        """
        带阻滤波器（阻止特定频率范围）

        Args:
            image: 输入图像
            cutoff_low: 低截止频率
            cutoff_high: 高截止频率
            filter_type: 滤波器类型
            order: 巴特沃斯阶数

        Returns:
            np.ndarray: 滤波后的图像
        """
        # 带阻 = 1 - 带通
        # FFT变换
        _, fft_shifted = self.fft_transform(image)

        # 创建带通掩码
        H, W = image.shape
        mask_high = self._create_circular_mask(H, W, cutoff_low, filter_type, 'high', order)
        mask_low = self._create_circular_mask(H, W, cutoff_high, filter_type, 'low', order)
        mask_bandpass = mask_high * mask_low

        # 带阻掩码
        mask = 1.0 - mask_bandpass

        # 应用滤波器
        filtered_fft = fft_shifted * mask

        # 逆变换
        result = self.ifft_transform(filtered_fft)

        return result.astype(np.uint8)

    # ==================== 核心函数：创建滤波掩码 ====================

    def _create_circular_mask(self, H: int, W: int,
                             cutoff: float,
                             filter_type: str = 'ideal',
                             pass_type: str = 'low',
                             order: int = 2) -> np.ndarray:
        """
        创建圆形滤波掩码

        Args:
            H: 图像高度
            W: 图像宽度
            cutoff: 截止频率
            filter_type: 滤波器类型 ('ideal', 'butterworth', 'gaussian')
            pass_type: 通过类型 ('low', 'high')
            order: 巴特沃斯阶数

        Returns:
            np.ndarray: 滤波掩码
        """
        # 创建距离矩阵（到中心的距离）
        center_u, center_v = H // 2, W // 2
        u = np.arange(H) - center_u
        v = np.arange(W) - center_v
        U, V = np.meshgrid(v, u)
        D = np.sqrt(U**2 + V**2)

        # 根据滤波器类型创建掩码
        if filter_type == 'ideal':
            # 理想滤波器
            if pass_type == 'low':
                mask = np.where(D <= cutoff, 1.0, 0.0)
            else:  # high
                mask = np.where(D > cutoff, 1.0, 0.0)

        elif filter_type == 'butterworth':
            # 巴特沃斯滤波器
            with np.errstate(divide='ignore', invalid='ignore'):
                if pass_type == 'low':
                    # H(u,v) = 1 / (1 + (D/D0)^(2n))
                    mask = 1.0 / (1.0 + (D / (cutoff + 1e-8))**(2 * order))
                else:  # high
                    # H(u,v) = 1 / (1 + (D0/D)^(2n))
                    mask = 1.0 / (1.0 + (cutoff / (D + 1e-8))**(2 * order))
            mask = np.nan_to_num(mask)

        elif filter_type == 'gaussian':
            # 高斯滤波器
            if pass_type == 'low':
                # H(u,v) = exp(-(D^2)/(2*D0^2))
                mask = np.exp(-(D**2) / (2 * cutoff**2))
            else:  # high
                # H(u,v) = 1 - exp(-(D^2)/(2*D0^2))
                mask = 1.0 - np.exp(-(D**2) / (2 * cutoff**2))

        else:
            raise ValueError(f"未知的滤波器类型: {filter_type}")

        return mask

    # ==================== 可视化 ====================

    def visualize_filters_comparison(self, image: np.ndarray,
                                     cutoff: float,
                                     save_path: Optional[str] = None):
        """
        可视化所有低通滤波器的效果对比

        Args:
            image: 输入图像
            cutoff: 截止频率
            save_path: 保存路径（可选）
        """
        # 应用各种滤波器
        ideal_lp = self.ideal_lowpass_filter(image, cutoff)
        butterworth_lp = self.butterworth_lowpass_filter(image, cutoff, order=2)
        gaussian_lp = self.gaussian_lowpass_filter(image, cutoff)

        ideal_hp = self.ideal_highpass_filter(image, cutoff)
        butterworth_hp = self.butterworth_highpass_filter(image, cutoff, order=2)
        gaussian_hp = self.gaussian_highpass_filter(image, cutoff)

        # 获取频谱
        _, fft_original = self.fft_transform(image)
        _, fft_ideal_lp = self.fft_transform(ideal_lp)
        _, fft_butterworth_lp = self.fft_transform(butterworth_lp)
        _, fft_gaussian_lp = self.fft_transform(gaussian_lp)

        # 创建掩码用于显示
        H, W = image.shape
        mask_ideal = self._create_circular_mask(H, W, cutoff, 'ideal', 'low')
        mask_butterworth = self._create_circular_mask(H, W, cutoff, 'butterworth', 'low', 2)
        mask_gaussian = self._create_circular_mask(H, W, cutoff, 'gaussian', 'low')

        # 显示
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle(f'频域滤波器效果对比 (截止频率={cutoff})', fontsize=16, fontweight='bold')

        # 第一行：原始图像和掩码
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(mask_ideal, cmap='gray')
        axes[0, 1].set_title('理想滤波器掩码')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(mask_butterworth, cmap='gray')
        axes[0, 2].set_title('巴特沃斯滤波器掩码')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(mask_gaussian, cmap='gray')
        axes[0, 3].set_title('高斯滤波器掩码')
        axes[0, 3].axis('off')

        # 第二行：低通滤波结果
        axes[1, 0].imshow(np.log(1 + np.abs(fft_original)), cmap='gray')
        axes[1, 0].set_title('原始频谱')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(ideal_lp, cmap='gray')
        axes[1, 1].set_title('理想低通结果')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(butterworth_lp, cmap='gray')
        axes[1, 2].set_title('巴特沃斯低通结果')
        axes[1, 2].axis('off')

        axes[1, 3].imshow(gaussian_lp, cmap='gray')
        axes[1, 3].set_title('高斯低通结果')
        axes[1, 3].axis('off')

        # 第三行：低通滤波后的频谱
        axes[2, 0].imshow(np.log(1 + np.abs(fft_ideal_lp)), cmap='gray')
        axes[2, 0].set_title('理想低通频谱')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(np.log(1 + np.abs(fft_butterworth_lp)), cmap='gray')
        axes[2, 1].set_title('巴特沃斯低通频谱')
        axes[2, 1].axis('off')

        axes[2, 2].imshow(np.log(1 + np.abs(fft_gaussian_lp)), cmap='gray')
        axes[2, 2].set_title('高斯低通频谱')
        axes[2, 2].axis('off')

        axes[2, 3].axis('off')

        # 第四行：高通滤波结果
        axes[3, 0].axis('off')

        axes[3, 1].imshow(ideal_hp, cmap='gray')
        axes[3, 1].set_title('理想高通结果')
        axes[3, 1].axis('off')

        axes[3, 2].imshow(butterworth_hp, cmap='gray')
        axes[3, 2].set_title('巴特沃斯高通结果')
        axes[3, 2].axis('off')

        axes[3, 3].imshow(gaussian_hp, cmap='gray')
        axes[3, 3].set_title('高斯高通结果')
        axes[3, 3].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"滤波对比图已保存到: {save_path}")

        plt.close()

    def visualize_cutoff_comparison(self, image: np.ndarray,
                                    cutoffs: list,
                                    filter_type: str = 'butterworth',
                                    save_path: Optional[str] = None):
        """
        可视化不同截止频率的效果

        Args:
            image: 输入图像
            cutoffs: 截止频率列表
            filter_type: 滤波器类型
            save_path: 保存路径（可选）
        """
        n_cutoffs = len(cutoffs)
        fig, axes = plt.subplots(2, n_cutoffs + 1, figsize=(4 * (n_cutoffs + 1), 8))
        fig.suptitle(f'{filter_type.capitalize()}滤波器 - 不同截止频率对比',
                    fontsize=16, fontweight='bold')

        # 显示原始图像
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')

        axes[1, 0].imshow(image, cmap='gray')
        axes[1, 0].set_title('原始图像')
        axes[1, 0].axis('off')

        # 对每个截止频率
        for idx, cutoff in enumerate(cutoffs, 1):
            # 低通滤波
            if filter_type == 'ideal':
                lp_result = self.ideal_lowpass_filter(image, cutoff)
                hp_result = self.ideal_highpass_filter(image, cutoff)
            elif filter_type == 'butterworth':
                lp_result = self.butterworth_lowpass_filter(image, cutoff, order=2)
                hp_result = self.butterworth_highpass_filter(image, cutoff, order=2)
            else:  # gaussian
                lp_result = self.gaussian_lowpass_filter(image, cutoff)
                hp_result = self.gaussian_highpass_filter(image, cutoff)

            # 显示低通结果
            axes[0, idx].imshow(lp_result, cmap='gray')
            axes[0, idx].set_title(f'低通 D0={cutoff}')
            axes[0, idx].axis('off')

            # 显示高通结果
            axes[1, idx].imshow(hp_result, cmap='gray')
            axes[1, idx].set_title(f'高通 D0={cutoff}')
            axes[1, idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"截止频率对比图已保存到: {save_path}")

        plt.close()


def main():
    """主函数：演示频域低通/高通滤波"""

    # 创建滤波器对象
    filter_obj = FrequencyLowpassFilter()

    # 创建测试图像
    print("创建测试图像...")
    H, W = 256, 256
    test_image = np.zeros((H, W), dtype=np.uint8)

    # 添加一些内容（低频）
    cv2.circle(test_image, (128, 128), 60, 200, -1)
    cv2.rectangle(test_image, (50, 50), (100, 100), 150, -1)

    # 添加细节（高频）
    for i in range(10, 240, 10):
        cv2.line(test_image, (i, 150), (i, 180), 255, 1)

    # 保存测试图像
    cv2.imwrite('lowpass_filter_test.png', test_image)
    print("测试图像已创建")

    # 可视化不同滤波器的效果
    print("\n比较不同类型的滤波器...")
    filter_obj.visualize_filters_comparison(
        test_image,
        cutoff=30,
        save_path='lowpass_filter_comparison.png'
    )

    # 可视化不同截止频率的效果
    print("\n比较不同截止频率...")
    filter_obj.visualize_cutoff_comparison(
        test_image,
        cutoffs=[10, 20, 40, 80],
        filter_type='butterworth',
        save_path='lowpass_filter_cutoffs.png'
    )

    # 演示带通/带阻滤波
    print("\n演示带通/带阻滤波...")
    bandpass_result = filter_obj.bandpass_filter(test_image, 20, 60)
    bandreject_result = filter_obj.bandreject_filter(test_image, 20, 60)

    cv2.imwrite('lowpass_filter_bandpass.png', bandpass_result)
    cv2.imwrite('lowpass_filter_bandreject.png', bandreject_result)

    print("\n频域滤波演示完成！")


if __name__ == "__main__":
    main()
