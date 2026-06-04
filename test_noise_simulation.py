"""
噪声模拟测试
"""

import pytest
import numpy as np
import cv2
from noise_simulation import NoiseSimulation


class TestNoiseSimulation:
    """测试噪声模拟类"""

    @pytest.fixture
    def noise_sim(self):
        """创建噪声模拟对象"""
        return NoiseSimulation()

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (50, 50), 30, 128, -1)
        return image

    # ==================== 加性噪声测试 ====================

    def test_gaussian_noise(self, noise_sim, test_image):
        """测试高斯噪声"""
        noisy = noise_sim.add_gaussian_noise(test_image, mean=0, sigma=25)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8
        assert not np.array_equal(noisy, test_image)  # 确实添加了噪声

    def test_uniform_noise(self, noise_sim, test_image):
        """测试均匀噪声"""
        noisy = noise_sim.add_uniform_noise(test_image, low=-20, high=20)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

    # ==================== 脉冲噪声测试 ====================

    def test_salt_pepper_noise(self, noise_sim, test_image):
        """测试椒盐噪声"""
        noisy = noise_sim.add_salt_pepper_noise(test_image, prob=0.05)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

        # 检查是否有0（椒）和255（盐）
        assert np.any(noisy == 0) or np.any(noisy == 255)

    def test_salt_pepper_noise_probability(self, noise_sim):
        """测试椒盐噪声概率"""
        # 使用灰度值为128的均匀图像，便于统计
        test_img = np.ones((100, 100), dtype=np.uint8) * 128
        prob = 0.1
        noisy = noise_sim.add_salt_pepper_noise(test_img, prob=prob)

        # 计算噪声像素数量（0或255）
        noise_pixels = np.sum((noisy == 0) | (noisy == 255))
        total_pixels = test_img.size

        # 实际概率应该接近设定概率（允许30%误差，因为随机性）
        actual_prob = noise_pixels / total_pixels
        assert abs(actual_prob - prob) < prob * 0.3

    def test_random_valued_noise(self, noise_sim, test_image):
        """测试随机值脉冲噪声"""
        noisy = noise_sim.add_random_valued_noise(test_image, prob=0.05)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

    # ==================== 乘性噪声测试 ====================

    def test_multiplicative_noise(self, noise_sim, test_image):
        """测试乘性噪声"""
        noisy = noise_sim.add_multiplicative_noise(test_image, mean=1.0, sigma=0.1)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

    def test_speckle_noise(self, noise_sim, test_image):
        """测试斑点噪声"""
        noisy = noise_sim.add_speckle_noise(test_image, variance=0.05)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

    # ==================== 周期性噪声测试 ====================

    def test_periodic_noise(self, noise_sim, test_image):
        """测试周期性噪声"""
        frequencies = [(10, 0), (0, 10)]
        amplitudes = [20, 20]

        noisy = noise_sim.add_periodic_noise(test_image, frequencies, amplitudes)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

    def test_periodic_noise_with_phase(self, noise_sim, test_image):
        """测试带相位的周期性噪声"""
        frequencies = [(10, 0)]
        amplitudes = [20]
        phases = [np.pi / 4]

        noisy = noise_sim.add_periodic_noise(test_image, frequencies, amplitudes, phases)

        assert noisy.shape == test_image.shape

    def test_sinusoidal_interference_horizontal(self, noise_sim, test_image):
        """测试水平正弦干扰"""
        noisy = noise_sim.add_sinusoidal_interference(
            test_image,
            direction='horizontal',
            frequency=20,
            amplitude=30
        )

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

    def test_sinusoidal_interference_vertical(self, noise_sim, test_image):
        """测试垂直正弦干扰"""
        noisy = noise_sim.add_sinusoidal_interference(
            test_image,
            direction='vertical',
            frequency=20,
            amplitude=30
        )

        assert noisy.shape == test_image.shape

    # ==================== 统计分布噪声测试 ====================

    def test_poisson_noise(self, noise_sim, test_image):
        """测试泊松噪声"""
        noisy = noise_sim.add_poisson_noise(test_image, scale=1.0)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

    def test_rayleigh_noise(self, noise_sim, test_image):
        """测试瑞利噪声"""
        noisy = noise_sim.add_rayleigh_noise(test_image, scale=30)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

    def test_exponential_noise(self, noise_sim, test_image):
        """测试指数噪声"""
        noisy = noise_sim.add_exponential_noise(test_image, scale=20)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

    def test_gamma_noise(self, noise_sim, test_image):
        """测试伽马噪声"""
        noisy = noise_sim.add_gamma_noise(test_image, shape=2.0, scale=10.0)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

    # ==================== 组合噪声测试 ====================

    def test_mixed_noise(self, noise_sim, test_image):
        """测试混合噪声"""
        noisy = noise_sim.add_mixed_noise(test_image, gaussian_sigma=15, sp_prob=0.02)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

    # ==================== 边界情况测试 ====================

    def test_all_black_image(self, noise_sim):
        """测试全黑图像"""
        black = np.zeros((100, 100), dtype=np.uint8)
        noisy = noise_sim.add_gaussian_noise(black, sigma=25)

        assert noisy.shape == black.shape
        assert not np.array_equal(noisy, black)

    def test_all_white_image(self, noise_sim):
        """测试全白图像"""
        white = np.ones((100, 100), dtype=np.uint8) * 255
        noisy = noise_sim.add_gaussian_noise(white, sigma=25)

        assert noisy.shape == white.shape

    def test_clipping(self, noise_sim, test_image):
        """测试像素值截断"""
        # 添加强噪声
        noisy = noise_sim.add_gaussian_noise(test_image, mean=0, sigma=100)

        # 确保所有值在[0, 255]范围内
        assert np.all(noisy >= 0)
        assert np.all(noisy <= 255)

    # ==================== 彩色图像测试 ====================

    def test_periodic_noise_color_image(self, noise_sim):
        """测试彩色图像的周期性噪声"""
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[:, :, 0] = 100  # R通道
        color_image[:, :, 1] = 150  # G通道
        color_image[:, :, 2] = 200  # B通道

        noisy = noise_sim.add_periodic_noise(
            color_image,
            frequencies=[(10, 0)],
            amplitudes=[20]
        )

        assert noisy.shape == color_image.shape
        assert noisy.dtype == np.uint8

    # ==================== 信噪比测试 ====================

    def test_calculate_snr(self, noise_sim, test_image):
        """测试SNR计算"""
        noisy = noise_sim.add_gaussian_noise(test_image, sigma=25)
        snr = noise_sim.calculate_snr(test_image, noisy)

        assert isinstance(snr, float)
        assert snr > 0  # SNR应该为正值

    def test_calculate_psnr(self, noise_sim, test_image):
        """测试PSNR计算"""
        noisy = noise_sim.add_gaussian_noise(test_image, sigma=25)
        psnr = noise_sim.calculate_psnr(test_image, noisy)

        assert isinstance(psnr, float)
        assert psnr > 0

    def test_snr_zero_noise(self, noise_sim, test_image):
        """测试无噪声时的SNR（应该为无穷大）"""
        snr = noise_sim.calculate_snr(test_image, test_image)
        assert snr == float('inf')

    def test_psnr_zero_noise(self, noise_sim, test_image):
        """测试无噪声时的PSNR（应该为无穷大）"""
        psnr = noise_sim.calculate_psnr(test_image, test_image)
        assert psnr == float('inf')

    def test_snr_increases_with_less_noise(self, noise_sim, test_image):
        """测试噪声减少时SNR增大"""
        noisy_weak = noise_sim.add_gaussian_noise(test_image, sigma=10)
        noisy_strong = noise_sim.add_gaussian_noise(test_image, sigma=50)

        snr_weak = noise_sim.calculate_snr(test_image, noisy_weak)
        snr_strong = noise_sim.calculate_snr(test_image, noisy_strong)

        assert snr_weak > snr_strong  # 弱噪声的SNR更高

    # ==================== 图像加载测试 ====================

    def test_load_image_success(self, noise_sim, test_image, tmp_path):
        """测试成功加载图像"""
        # 创建临时图像文件
        temp_file = tmp_path / "test.png"
        cv2.imwrite(str(temp_file), test_image)

        success = noise_sim.load_image(str(temp_file))
        assert success
        assert noise_sim.image is not None

    def test_load_image_failure(self, noise_sim):
        """测试加载不存在的图像"""
        success = noise_sim.load_image("nonexistent.png")
        assert not success

    # ==================== 噪声估计测试 ====================

    def test_extract_noise(self, noise_sim, test_image):
        """测试噪声提取"""
        noisy = noise_sim.add_gaussian_noise(test_image, sigma=25)
        noise = noise_sim.extract_noise(test_image, noisy)

        assert noise.shape == test_image.shape
        assert noise.dtype == np.float32

    def test_estimate_noise_statistics(self, noise_sim, test_image):
        """测试噪声统计特征估计"""
        noisy = noise_sim.add_gaussian_noise(test_image, sigma=25)
        noise = noise_sim.extract_noise(test_image, noisy)
        stats = noise_sim.estimate_noise_statistics(noise)

        # 检查所有统计量都存在
        assert 'mean' in stats
        assert 'std' in stats
        assert 'variance' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'range' in stats
        assert 'median' in stats

        # 检查统计量类型
        assert isinstance(stats['mean'], float)
        assert isinstance(stats['std'], float)

    def test_detect_impulse_noise(self, noise_sim, test_image):
        """测试脉冲噪声检测"""
        # 添加椒盐噪声
        noisy = noise_sim.add_salt_pepper_noise(test_image, prob=0.1)
        noise = noise_sim.extract_noise(test_image, noisy)

        impulse_ratio = noise_sim.detect_impulse_noise(noise, threshold=100)

        # 脉冲噪声比例应该大于0
        assert impulse_ratio > 0
        assert impulse_ratio <= 1.0

    def test_estimate_noise_type_gaussian(self, noise_sim, test_image):
        """测试高斯噪声类型估计"""
        noisy = noise_sim.add_gaussian_noise(test_image, sigma=25)
        result = noise_sim.estimate_noise_type(test_image, noisy)

        assert 'primary_type' in result
        assert 'confidence' in result
        assert 'statistics' in result
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1

    def test_estimate_noise_type_salt_pepper(self, noise_sim, test_image):
        """测试椒盐噪声类型估计"""
        noisy = noise_sim.add_salt_pepper_noise(test_image, prob=0.05)
        result = noise_sim.estimate_noise_type(test_image, noisy)

        # 应该识别出脉冲噪声
        assert '脉冲' in result['primary_type'] or '椒盐' in result['primary_type']

    def test_estimate_noise_type_with_region(self, noise_sim, test_image):
        """测试局部区域噪声估计"""
        noisy = noise_sim.add_gaussian_noise(test_image, sigma=25)
        region = (20, 80, 20, 80)
        result = noise_sim.estimate_noise_type(test_image, noisy, region=region)

        assert 'primary_type' in result
        assert 'confidence' in result

    def test_estimate_noise_statistics_zero_std(self, noise_sim):
        """测试零标准差情况"""
        # 无噪声情况
        test_img = np.ones((50, 50), dtype=np.uint8) * 100
        noise = np.zeros((50, 50), dtype=np.float32)

        stats = noise_sim.estimate_noise_statistics(noise)

        # 标准差为0时，偏度和峰度应该为0
        assert stats['std'] == 0
        assert stats['skewness'] == 0
        assert stats['kurtosis'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
