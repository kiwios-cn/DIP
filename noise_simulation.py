"""
噪声模拟模块

实现各种常见的图像噪声类型，用于测试去噪算法。
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class NoiseSimulation:
    """图像噪声模拟类"""

    def __init__(self):
        self.image = None

    # ==================== 加性噪声 ====================

    def add_gaussian_noise(self, image: np.ndarray,
                          mean: float = 0,
                          sigma: float = 25) -> np.ndarray:
        """
        添加高斯噪声（正态分布噪声）

        Args:
            image: 输入图像
            mean: 噪声均值，默认0
            sigma: 噪声标准差，默认25

        Returns:
            np.ndarray: 添加噪声后的图像

        应用场景：
            - 模拟传感器噪声
            - 模拟热噪声
            - 模拟量化噪声
        """
        noisy = image.copy().astype(np.float32)
        noise = np.random.normal(mean, sigma, image.shape)
        noisy = noisy + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_uniform_noise(self, image: np.ndarray,
                         low: float = -20,
                         high: float = 20) -> np.ndarray:
        """
        添加均匀噪声

        Args:
            image: 输入图像
            low: 噪声下界
            high: 噪声上界

        Returns:
            np.ndarray: 添加噪声后的图像
        """
        noisy = image.copy().astype(np.float32)
        noise = np.random.uniform(low, high, image.shape)
        noisy = noisy + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # ==================== 脉冲噪声 ====================

    def add_salt_pepper_noise(self, image: np.ndarray,
                             prob: float = 0.05,
                             salt_prob: float = 0.5) -> np.ndarray:
        """
        添加椒盐噪声（脉冲噪声）

        Args:
            image: 输入图像
            prob: 噪声总概率，默认0.05
            salt_prob: 盐噪声（白点）在总噪声中的比例，默认0.5

        Returns:
            np.ndarray: 添加噪声后的图像

        应用场景：
            - 模拟传输错误
            - 模拟模数转换错误
            - 模拟坏点
        """
        noisy = image.copy()

        # 生成噪声掩码
        noise_mask = np.random.random(image.shape) < prob

        # 在噪声位置中随机分配盐和椒
        noise_positions = noise_mask.nonzero()
        n_noise = len(noise_positions[0])

        if n_noise > 0:
            # 决定哪些是盐（白点），哪些是椒（黑点）
            salt_positions = np.random.random(n_noise) < salt_prob

            # 应用盐噪声
            for i in range(n_noise):
                if salt_positions[i]:
                    noisy[noise_positions[0][i], noise_positions[1][i]] = 255
                else:
                    noisy[noise_positions[0][i], noise_positions[1][i]] = 0

        return noisy

    def add_random_valued_noise(self, image: np.ndarray,
                               prob: float = 0.05) -> np.ndarray:
        """
        添加随机值脉冲噪声

        与椒盐噪声不同，噪声像素值是随机的，不仅限于0和255

        Args:
            image: 输入图像
            prob: 噪声概率

        Returns:
            np.ndarray: 添加噪声后的图像
        """
        noisy = image.copy()
        noise_mask = np.random.random(image.shape) < prob
        noisy[noise_mask] = np.random.randint(0, 256, np.sum(noise_mask))
        return noisy

    # ==================== 乘性噪声 ====================

    def add_multiplicative_noise(self, image: np.ndarray,
                                mean: float = 1.0,
                                sigma: float = 0.1) -> np.ndarray:
        """
        添加乘性噪声（也称为斑点噪声）

        噪声与信号强度成正比，常见于雷达、超声等成像

        Args:
            image: 输入图像
            mean: 噪声均值，默认1.0
            sigma: 噪声标准差，默认0.1

        Returns:
            np.ndarray: 添加噪声后的图像

        应用场景：
            - SAR图像噪声
            - 超声图像噪声
            - 雷达图像噪声
        """
        noisy = image.copy().astype(np.float32)
        noise = np.random.normal(mean, sigma, image.shape)
        noisy = noisy * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_speckle_noise(self, image: np.ndarray,
                         variance: float = 0.05) -> np.ndarray:
        """
        添加斑点噪声（乘性噪声的一种）

        公式: I_noisy = I + I * noise

        Args:
            image: 输入图像
            variance: 噪声方差

        Returns:
            np.ndarray: 添加噪声后的图像
        """
        noisy = image.copy().astype(np.float32)
        noise = np.random.randn(*image.shape) * np.sqrt(variance)
        noisy = noisy + noisy * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # ==================== 周期性噪声 ====================

    def add_periodic_noise(self, image: np.ndarray,
                          frequencies: list,
                          amplitudes: list,
                          phases: Optional[list] = None) -> np.ndarray:
        """
        添加周期性噪声（正弦噪声）

        Args:
            image: 输入图像
            frequencies: 频率列表 [(freq_u, freq_v), ...]
            amplitudes: 幅度列表
            phases: 相位列表（可选），默认为0

        Returns:
            np.ndarray: 添加噪声后的图像

        应用场景：
            - 电磁干扰
            - 扫描线干扰
            - 设备振动
        """
        H, W = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]
        noisy = image.copy().astype(np.float32)

        if phases is None:
            phases = [0] * len(frequencies)

        y = np.arange(H)
        x = np.arange(W)
        X, Y = np.meshgrid(x, y)

        for (freq_u, freq_v), amp, phase in zip(frequencies, amplitudes, phases):
            noise = amp * np.sin(2 * np.pi * (freq_u * Y / H + freq_v * X / W) + phase)

            if len(image.shape) == 3:
                noise = noise[:, :, np.newaxis]

            noisy = noisy + noise

        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_sinusoidal_interference(self, image: np.ndarray,
                                   direction: str = 'horizontal',
                                   frequency: float = 20,
                                   amplitude: float = 30) -> np.ndarray:
        """
        添加正弦干扰条纹

        Args:
            image: 输入图像
            direction: 方向 'horizontal' 或 'vertical'
            frequency: 频率
            amplitude: 幅度

        Returns:
            np.ndarray: 添加噪声后的图像
        """
        if direction == 'horizontal':
            freq_u, freq_v = frequency, 0
        else:  # vertical
            freq_u, freq_v = 0, frequency

        return self.add_periodic_noise(image, [(freq_u, freq_v)], [amplitude])

    # ==================== 泊松噪声 ====================

    def add_poisson_noise(self, image: np.ndarray,
                         scale: float = 1.0) -> np.ndarray:
        """
        添加泊松噪声（光子计数噪声）

        泊松噪声是光子到达传感器时固有的统计噪声

        Args:
            image: 输入图像
            scale: 缩放因子，控制噪声强度

        Returns:
            np.ndarray: 添加噪声后的图像

        应用场景：
            - 低光照成像
            - 医学成像
            - 天文成像
        """
        # 归一化到[0, 1]
        normalized = image.astype(np.float32) / 255.0

        # 缩放以控制噪声强度
        scaled = normalized * scale

        # 添加泊松噪声
        noisy = np.random.poisson(scaled * 255) / scale

        return np.clip(noisy, 0, 255).astype(np.uint8)

    # ==================== 瑞利噪声 ====================

    def add_rayleigh_noise(self, image: np.ndarray,
                          scale: float = 30) -> np.ndarray:
        """
        添加瑞利噪声

        Args:
            image: 输入图像
            scale: 尺度参数

        Returns:
            np.ndarray: 添加噪声后的图像

        应用场景：
            - 雷达图像
            - 声纳图像
        """
        noisy = image.copy().astype(np.float32)
        noise = np.random.rayleigh(scale, image.shape)
        noisy = noisy + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # ==================== 指数噪声 ====================

    def add_exponential_noise(self, image: np.ndarray,
                             scale: float = 20) -> np.ndarray:
        """
        添加指数噪声

        Args:
            image: 输入图像
            scale: 尺度参数（均值的倒数）

        Returns:
            np.ndarray: 添加噪声后的图像
        """
        noisy = image.copy().astype(np.float32)
        noise = np.random.exponential(scale, image.shape)
        noisy = noisy + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # ==================== 伽马噪声 ====================

    def add_gamma_noise(self, image: np.ndarray,
                       shape: float = 2.0,
                       scale: float = 10.0) -> np.ndarray:
        """
        添加伽马噪声（厄朗噪声）

        Args:
            image: 输入图像
            shape: 形状参数（k）
            scale: 尺度参数（θ）

        Returns:
            np.ndarray: 添加噪声后的图像
        """
        noisy = image.copy().astype(np.float32)
        noise = np.random.gamma(shape, scale, image.shape)
        noisy = noisy + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # ==================== 组合噪声 ====================

    def add_mixed_noise(self, image: np.ndarray,
                       gaussian_sigma: float = 15,
                       sp_prob: float = 0.02) -> np.ndarray:
        """
        添加混合噪声（高斯 + 椒盐）

        模拟真实场景中多种噪声共存的情况

        Args:
            image: 输入图像
            gaussian_sigma: 高斯噪声标准差
            sp_prob: 椒盐噪声概率

        Returns:
            np.ndarray: 添加噪声后的图像
        """
        # 先添加高斯噪声
        noisy = self.add_gaussian_noise(image, sigma=gaussian_sigma)
        # 再添加椒盐噪声
        noisy = self.add_salt_pepper_noise(noisy, prob=sp_prob)
        return noisy

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

    def calculate_snr(self, original: np.ndarray, noisy: np.ndarray) -> float:
        """
        计算信噪比（SNR）

        SNR = 10 * log10(signal_power / noise_power)

        Args:
            original: 原始图像
            noisy: 含噪图像

        Returns:
            float: 信噪比（dB）
        """
        signal = original.astype(np.float32)
        noise = (noisy.astype(np.float32) - signal)

        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power == 0:
            return float('inf')

        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

    def calculate_psnr(self, original: np.ndarray, noisy: np.ndarray) -> float:
        """
        计算峰值信噪比（PSNR）

        PSNR = 10 * log10(MAX^2 / MSE)

        Args:
            original: 原始图像
            noisy: 含噪图像

        Returns:
            float: 峰值信噪比（dB）
        """
        mse = np.mean((original.astype(np.float32) - noisy.astype(np.float32)) ** 2)

        if mse == 0:
            return float('inf')

        max_pixel = 255.0
        psnr = 10 * np.log10(max_pixel ** 2 / mse)
        return float(psnr)

    # ==================== 噪声估计 ====================

    def extract_noise(self, original: np.ndarray, noisy: np.ndarray) -> np.ndarray:
        """
        从含噪图像中提取噪声

        Args:
            original: 原始图像
            noisy: 含噪图像

        Returns:
            np.ndarray: 噪声图像
        """
        return noisy.astype(np.float32) - original.astype(np.float32)

    def estimate_noise_statistics(self, noise: np.ndarray) -> dict:
        """
        估计噪声的统计特征

        Args:
            noise: 噪声图像（已提取的纯噪声）

        Returns:
            dict: 包含各种统计特征的字典
        """
        # 基本统计量
        mean = np.mean(noise)
        std = np.std(noise)
        variance = np.var(noise)

        # 高阶统计量
        skewness = np.mean(((noise - mean) / std) ** 3) if std > 0 else 0
        kurtosis = np.mean(((noise - mean) / std) ** 4) if std > 0 else 0

        # 范围统计
        min_val = np.min(noise)
        max_val = np.max(noise)
        range_val = max_val - min_val

        # 中位数和四分位数
        median = np.median(noise)
        q1 = np.percentile(noise, 25)
        q3 = np.percentile(noise, 75)
        iqr = q3 - q1

        return {
            'mean': float(mean),
            'std': float(std),
            'variance': float(variance),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'min': float(min_val),
            'max': float(max_val),
            'range': float(range_val),
            'median': float(median),
            'q1': float(q1),
            'q3': float(q3),
            'iqr': float(iqr)
        }

    def detect_impulse_noise(self, noise: np.ndarray, threshold: float = 100) -> float:
        """
        检测脉冲噪声（椒盐噪声）的比例

        Args:
            noise: 噪声图像
            threshold: 脉冲阈值

        Returns:
            float: 脉冲噪声比例
        """
        impulse_mask = np.abs(noise) > threshold
        impulse_ratio = np.sum(impulse_mask) / noise.size
        return float(impulse_ratio)

    def estimate_noise_type(self, original: np.ndarray, noisy: np.ndarray,
                           region: Optional[Tuple[int, int, int, int]] = None) -> dict:
        """
        估计噪声类型（基于统计特征）

        Args:
            original: 原始图像
            noisy: 含噪图像
            region: 可选的局部区域 (y1, y2, x1, x2)，用于局部噪声估计

        Returns:
            dict: 包含噪声类型判断和置信度的字典
        """
        # 如果指定了区域，只分析该区域
        if region is not None:
            y1, y2, x1, x2 = region
            original_roi = original[y1:y2, x1:x2]
            noisy_roi = noisy[y1:y2, x1:x2]
        else:
            original_roi = original
            noisy_roi = noisy

        # 提取噪声
        noise = self.extract_noise(original_roi, noisy_roi)

        # 计算统计特征
        stats = self.estimate_noise_statistics(noise)

        # 检测脉冲噪声
        impulse_ratio = self.detect_impulse_noise(noise, threshold=100)

        # 基于统计特征判断噪声类型
        noise_types = []
        confidences = []

        # 1. 检测脉冲噪声（椒盐噪声）- 优先级最高
        if impulse_ratio > 0.01:  # 超过1%的脉冲
            noise_types.append('脉冲噪声（椒盐噪声）')
            confidences.append(min(impulse_ratio * 100, 0.95))

        # 2. 检测指数噪声 - 特征最明显
        # 特征：强正偏度（约2），峰度约9
        if stats['skewness'] > 1.5 and stats['kurtosis'] > 6:
            exp_score = 0
            if stats['skewness'] > 1.8:
                exp_score += 0.5
            if stats['kurtosis'] > 7:
                exp_score += 0.4
            if stats['min'] >= -stats['std']:
                exp_score += 0.1

            if exp_score > 0.6:
                noise_types.append('指数噪声')
                confidences.append(min(exp_score, 0.95))

        # 3. 检测瑞利噪声
        # 特征：正偏度（约0.63），峰度约3.2
        if 0.3 < stats['skewness'] < 1.0 and 3.0 < stats['kurtosis'] < 3.8:
            rayleigh_score = 0
            if 0.4 < stats['skewness'] < 0.8:
                rayleigh_score += 0.5
            if 3.1 < stats['kurtosis'] < 3.4:
                rayleigh_score += 0.3
            if stats['mean'] > 0 and stats['min'] >= -stats['std']:
                rayleigh_score += 0.2

            if rayleigh_score > 0.5:
                noise_types.append('瑞利噪声')
                confidences.append(min(rayleigh_score, 0.95))

        # 4. 检测伽马噪声
        # 特征：中等正偏度（0.5-1.5），峰度（3.5-6）
        if 0.5 < stats['skewness'] < 1.5 and 3.5 < stats['kurtosis'] < 6.5:
            gamma_score = 0
            if 0.8 < stats['skewness'] < 1.3:
                gamma_score += 0.5
            if 4.0 < stats['kurtosis'] < 6.0:
                gamma_score += 0.3
            if stats['mean'] > 0:
                gamma_score += 0.2

            if gamma_score > 0.5:
                noise_types.append('伽马噪声')
                confidences.append(min(gamma_score, 0.9))

        # 5. 检测均匀噪声 - 峰度是关键
        # 特征：峰度约为1.8（远小于高斯的3），偏度接近0
        if 1.5 < stats['kurtosis'] < 2.3 and abs(stats['skewness']) < 0.5:
            uniform_score = 0
            # 峰度是最重要的特征
            kurtosis_diff = abs(stats['kurtosis'] - 1.8)
            if kurtosis_diff < 0.2:
                uniform_score += 0.7
            elif kurtosis_diff < 0.4:
                uniform_score += 0.5
            # 偏度接近0
            if abs(stats['skewness']) < 0.3:
                uniform_score += 0.2

            if uniform_score > 0.6:
                noise_types.append('均匀噪声')
                confidences.append(min(uniform_score, 0.9))

        # 6. 检测高斯噪声
        # 特征：偏度接近0，峰度接近3
        if abs(stats['skewness']) < 0.5 and 2.5 < stats['kurtosis'] < 3.8:
            gaussian_score = 0
            if abs(stats['skewness']) < 0.3:
                gaussian_score += 0.4
            if 2.8 < stats['kurtosis'] < 3.3:
                gaussian_score += 0.5
            if abs(stats['mean']) < stats['std'] * 0.3:
                gaussian_score += 0.1

            if gaussian_score > 0.6:
                noise_types.append('高斯噪声')
                confidences.append(min(gaussian_score, 0.9))

        # 如果没有识别出任何噪声类型，默认为高斯噪声（最常见）
        if len(noise_types) == 0:
            if abs(stats['skewness']) < 1.0 and 2.0 < stats['kurtosis'] < 5.0:
                noise_types.append('高斯噪声')
                confidences.append(0.5)
            else:
                noise_types.append('未知噪声类型')
                confidences.append(0.0)

        # 返回置信度最高的结果
        max_idx = np.argmax(confidences)

        return {
            'primary_type': noise_types[max_idx],
            'confidence': confidences[max_idx],
            'all_candidates': list(zip(noise_types, confidences)),
            'statistics': stats,
            'impulse_ratio': impulse_ratio
        }

    def visualize_noise_estimation(self, original: np.ndarray, noisy: np.ndarray,
                                   region: Optional[Tuple[int, int, int, int]] = None,
                                   save_path: Optional[str] = None):
        """
        可视化噪声估计结果

        Args:
            original: 原始图像
            noisy: 含噪图像
            region: 可选的局部区域
            save_path: 保存路径
        """
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 估计噪声类型
        result = self.estimate_noise_type(original, noisy, region)

        # 提取噪声
        if region is not None:
            y1, y2, x1, x2 = region
            noise = self.extract_noise(original[y1:y2, x1:x2], noisy[y1:y2, x1:x2])
        else:
            noise = self.extract_noise(original, noisy)

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 原始图像
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original, cmap='gray')
        ax1.set_title('原始图像', fontsize=12)
        ax1.axis('off')

        # 2. 含噪图像
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(noisy, cmap='gray')
        if region is not None:
            y1, y2, x1, x2 = region
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                fill=False, color='red', linewidth=2)
            ax2.add_patch(rect)
            ax2.set_title('含噪图像（红框为分析区域）', fontsize=12)
        else:
            ax2.set_title('含噪图像', fontsize=12)
        ax2.axis('off')

        # 3. 噪声图像
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(noise, cmap='gray')
        ax3.set_title('提取的噪声', fontsize=12)
        ax3.axis('off')

        # 4. 噪声直方图
        ax4 = fig.add_subplot(gs[1, :])
        ax4.hist(noise.flatten(), bins=100, density=True, alpha=0.7, color='blue')
        ax4.set_xlabel('噪声值', fontsize=11)
        ax4.set_ylabel('概率密度', fontsize=11)
        ax4.set_title('噪声分布直方图', fontsize=12)
        ax4.grid(True, alpha=0.3)

        # 5. 统计信息
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        stats = result['statistics']
        stats_text = f"""统计特征：
均值: {stats['mean']:.3f}
标准差: {stats['std']:.3f}
方差: {stats['variance']:.3f}
偏度: {stats['skewness']:.3f}
峰度: {stats['kurtosis']:.3f}
范围: [{stats['min']:.1f}, {stats['max']:.1f}]
中位数: {stats['median']:.3f}
脉冲比例: {result['impulse_ratio']:.3%}"""
        ax5.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 6. 噪声类型判断
        ax6 = fig.add_subplot(gs[2, 1:])
        ax6.axis('off')

        result_text = f"""噪声类型估计结果：

主要类型: {result['primary_type']}
置信度: {result['confidence']:.1%}

所有候选类型:
"""
        for noise_type, conf in result['all_candidates']:
            result_text += f"  • {noise_type}: {conf:.1%}\n"

        ax6.text(0.1, 0.5, result_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"噪声估计结果已保存到: {save_path}")

        plt.show()

    # ==================== 可视化 ====================

    def visualize_all_noises(self, image: np.ndarray,
                            save_path: Optional[str] = None):
        """
        可视化所有噪声类型

        Args:
            image: 输入图像
            save_path: 保存路径（可选）
        """
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()

        # 原图
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')

        # 各种噪声
        noises = [
            ('高斯噪声\n(σ=25)', self.add_gaussian_noise(image, sigma=25)),
            ('均匀噪声\n(±20)', self.add_uniform_noise(image, -20, 20)),
            ('椒盐噪声\n(p=0.05)', self.add_salt_pepper_noise(image, prob=0.05)),
            ('随机值噪声\n(p=0.05)', self.add_random_valued_noise(image, prob=0.05)),
            ('乘性噪声\n(σ=0.1)', self.add_multiplicative_noise(image, sigma=0.1)),
            ('斑点噪声\n(var=0.05)', self.add_speckle_noise(image, variance=0.05)),
            ('周期噪声\n(freq=20)', self.add_sinusoidal_interference(image, 'horizontal', 20, 30)),
            ('泊松噪声', self.add_poisson_noise(image, scale=1.0)),
            ('瑞利噪声\n(scale=30)', self.add_rayleigh_noise(image, scale=30)),
            ('指数噪声\n(scale=20)', self.add_exponential_noise(image, scale=20)),
            ('伽马噪声\n(k=2, θ=10)', self.add_gamma_noise(image, shape=2, scale=10)),
            ('混合噪声\n(高斯+椒盐)', self.add_mixed_noise(image)),
        ]

        for idx, (title, noisy) in enumerate(noises, start=1):
            axes[idx].imshow(noisy, cmap='gray')
            snr = self.calculate_snr(image, noisy)
            psnr = self.calculate_psnr(image, noisy)
            axes[idx].set_title(f'{title}\nSNR={snr:.1f}dB\nPSNR={psnr:.1f}dB',
                              fontsize=10)
            axes[idx].axis('off')

        # 隐藏多余的子图
        for idx in range(len(noises) + 1, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")

        plt.show()


def main():
    """演示程序"""
    print("噪声模拟与估计演示程序")
    print("=" * 50)

    # 创建噪声模拟对象
    noise_sim = NoiseSimulation()

    # 创建测试图像
    test_image = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(test_image, (128, 128), 80, 200, -1)
    cv2.rectangle(test_image, (50, 50), (100, 100), 150, -1)
    cv2.imwrite('noise_test_image.png', test_image)
    print("已创建测试图像: noise_test_image.png")

    # 可视化所有噪声
    print("\n步骤1: 生成所有噪声类型对比图...")
    noise_sim.visualize_all_noises(test_image, save_path='noise_comparison.png')

    # 演示噪声估计功能
    print("\n步骤2: 演示噪声估计功能...")

    # 测试不同类型的噪声
    noise_examples = [
        ('高斯噪声', noise_sim.add_gaussian_noise(test_image, sigma=25)),
        ('椒盐噪声', noise_sim.add_salt_pepper_noise(test_image, prob=0.05)),
        ('瑞利噪声', noise_sim.add_rayleigh_noise(test_image, scale=30)),
        ('指数噪声', noise_sim.add_exponential_noise(test_image, scale=20)),
        ('伽马噪声', noise_sim.add_gamma_noise(test_image, shape=2, scale=10)),
        ('均匀噪声', noise_sim.add_uniform_noise(test_image, -20, 20)),
    ]

    print("\n噪声类型识别结果:")
    print("-" * 70)

    for idx, (true_type, noisy_img) in enumerate(noise_examples, 1):
        result = noise_sim.estimate_noise_type(test_image, noisy_img)
        print(f"{idx}. 真实类型: {true_type:12s} | "
              f"估计类型: {result['primary_type']:20s} | "
              f"置信度: {result['confidence']:.1%}")

        # 保存第一个噪声估计的可视化结果
        if idx == 1:
            noise_sim.visualize_noise_estimation(
                test_image, noisy_img,
                save_path='noise_estimation_example.png'
            )

    # 演示局部噪声估计
    print("\n步骤3: 演示局部区域噪声估计...")
    mixed_noisy = noise_sim.add_mixed_noise(test_image)
    region = (50, 150, 50, 150)  # 选择中心区域
    noise_sim.visualize_noise_estimation(
        test_image, mixed_noisy,
        region=region,
        save_path='noise_estimation_local.png'
    )

    print("\n✅ 演示完成！")
    print(f"\n生成文件:")
    print(f"  - noise_test_image.png (测试图像)")
    print(f"  - noise_comparison.png (所有噪声类型对比)")
    print(f"  - noise_estimation_example.png (高斯噪声估计示例)")
    print(f"  - noise_estimation_local.png (局部区域噪声估计)")


if __name__ == '__main__':
    main()
