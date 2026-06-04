"""
均值滤波器测试
"""

import pytest
import numpy as np
import cv2
from mean_filters import MeanFilters


class TestMeanFilters:
    """测试均值滤波器类"""

    @pytest.fixture
    def filters(self):
        """创建滤波器对象"""
        return MeanFilters()

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (50, 50), 30, 128, -1)
        return image

    @pytest.fixture
    def noisy_image(self, test_image):
        """创建含噪图像"""
        # 添加高斯噪声
        noise = np.random.normal(0, 10, test_image.shape)
        noisy = test_image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # ==================== 算术均值滤波器测试 ====================

    def test_arithmetic_mean_filter_basic(self, filters, test_image):
        """测试算术均值滤波器基本功能"""
        result = filters.arithmetic_mean_filter(test_image, kernel_size=3)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8
        assert not np.array_equal(result, test_image)  # 应该有变化

    def test_arithmetic_mean_filter_kernel_sizes(self, filters, test_image):
        """测试不同卷积核大小"""
        for size in [3, 5, 7]:
            result = filters.arithmetic_mean_filter(test_image, kernel_size=size)
            assert result.shape == test_image.shape

    def test_arithmetic_mean_filter_even_kernel(self, filters, test_image):
        """测试偶数卷积核应该抛出异常"""
        with pytest.raises(ValueError):
            filters.arithmetic_mean_filter(test_image, kernel_size=4)

    def test_arithmetic_mean_smoothing(self, filters):
        """测试算术均值滤波的平滑效果"""
        # 使用亮度较高的测试图像
        test_img = np.ones((100, 100), dtype=np.uint8) * 128
        cv2.circle(test_img, (50, 50), 30, 200, -1)

        # 添加高斯噪声
        noise = np.random.normal(0, 15, test_img.shape)
        noisy = test_img.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        filtered = filters.arithmetic_mean_filter(noisy, kernel_size=5)

        # 滤波后应该更接近原图
        mse_before = np.mean((noisy.astype(float) - test_img.astype(float)) ** 2)
        mse_after = np.mean((filtered.astype(float) - test_img.astype(float)) ** 2)

        assert mse_after < mse_before  # 滤波后误差应该减小

    # ==================== 几何均值滤波器测试 ====================

    def test_geometric_mean_filter_basic(self, filters, test_image):
        """测试几何均值滤波器基本功能"""
        result = filters.geometric_mean_filter(test_image, kernel_size=3)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_geometric_mean_filter_kernel_sizes(self, filters, test_image):
        """测试不同卷积核大小"""
        for size in [3, 5, 7]:
            result = filters.geometric_mean_filter(test_image, kernel_size=size)
            assert result.shape == test_image.shape

    def test_geometric_mean_filter_even_kernel(self, filters, test_image):
        """测试偶数卷积核应该抛出异常"""
        with pytest.raises(ValueError):
            filters.geometric_mean_filter(test_image, kernel_size=4)

    def test_geometric_mean_zero_handling(self, filters):
        """测试几何均值滤波器处理0值"""
        image_with_zeros = np.array([[0, 100, 0],
                                     [100, 200, 100],
                                     [0, 100, 0]], dtype=np.uint8)

        result = filters.geometric_mean_filter(image_with_zeros, kernel_size=3)
        assert result.shape == image_with_zeros.shape
        assert not np.any(np.isnan(result))  # 不应该有NaN

    # ==================== 谐波均值滤波器测试 ====================

    def test_harmonic_mean_filter_basic(self, filters, test_image):
        """测试谐波均值滤波器基本功能"""
        result = filters.harmonic_mean_filter(test_image, kernel_size=3)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_harmonic_mean_filter_kernel_sizes(self, filters, test_image):
        """测试不同卷积核大小"""
        for size in [3, 5, 7]:
            result = filters.harmonic_mean_filter(test_image, kernel_size=size)
            assert result.shape == test_image.shape

    def test_harmonic_mean_filter_even_kernel(self, filters, test_image):
        """测试偶数卷积核应该抛出异常"""
        with pytest.raises(ValueError):
            filters.harmonic_mean_filter(test_image, kernel_size=4)

    def test_harmonic_mean_zero_handling(self, filters):
        """测试谐波均值滤波器处理0值"""
        image_with_zeros = np.array([[0, 100, 0],
                                     [100, 200, 100],
                                     [0, 100, 0]], dtype=np.uint8)

        result = filters.harmonic_mean_filter(image_with_zeros, kernel_size=3)
        assert result.shape == image_with_zeros.shape
        assert not np.any(np.isnan(result))

    def test_harmonic_mean_salt_noise(self, filters, test_image):
        """测试谐波均值滤波器去除盐噪声"""
        # 添加盐噪声（白点）
        salt_noisy = test_image.copy()
        num_salt = 200
        coords = [np.random.randint(0, i, num_salt) for i in test_image.shape]
        salt_noisy[coords[0], coords[1]] = 255

        filtered = filters.harmonic_mean_filter(salt_noisy, kernel_size=5)

        # 检查白点是否被抑制
        assert np.sum(filtered == 255) < np.sum(salt_noisy == 255)

    # ==================== 逆谐波均值滤波器测试 ====================

    def test_contraharmonic_mean_filter_basic(self, filters, test_image):
        """测试逆谐波均值滤波器基本功能"""
        result = filters.contraharmonic_mean_filter(test_image, kernel_size=3, Q=1.0)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_contraharmonic_mean_filter_kernel_sizes(self, filters, test_image):
        """测试不同卷积核大小"""
        for size in [3, 5, 7]:
            result = filters.contraharmonic_mean_filter(test_image, kernel_size=size, Q=1.0)
            assert result.shape == test_image.shape

    def test_contraharmonic_mean_filter_even_kernel(self, filters, test_image):
        """测试偶数卷积核应该抛出异常"""
        with pytest.raises(ValueError):
            filters.contraharmonic_mean_filter(test_image, kernel_size=4, Q=1.0)

    def test_contraharmonic_Q_zero(self, filters, test_image):
        """测试Q=0时等同于算术均值"""
        arith = filters.arithmetic_mean_filter(test_image, kernel_size=3)
        contra = filters.contraharmonic_mean_filter(test_image, kernel_size=3, Q=0)

        # 结果应该非常接近
        diff = np.abs(arith.astype(float) - contra.astype(float))
        assert np.mean(diff) < 1.0  # 平均差异小于1

    def test_contraharmonic_positive_Q(self, filters, test_image):
        """测试正Q值去除椒噪声"""
        # 添加椒噪声（黑点）
        pepper_noisy = test_image.copy()
        num_pepper = 200
        coords = [np.random.randint(0, i, num_pepper) for i in test_image.shape]
        pepper_noisy[coords[0], coords[1]] = 0

        filtered = filters.contraharmonic_mean_filter(pepper_noisy, kernel_size=5, Q=1.5)

        # 检查黑点是否被抑制
        assert np.sum(filtered == 0) < np.sum(pepper_noisy == 0)

    def test_contraharmonic_negative_Q(self, filters, test_image):
        """测试负Q值去除盐噪声"""
        # 添加盐噪声（白点）
        salt_noisy = test_image.copy()
        num_salt = 200
        coords = [np.random.randint(0, i, num_salt) for i in test_image.shape]
        salt_noisy[coords[0], coords[1]] = 255

        filtered = filters.contraharmonic_mean_filter(salt_noisy, kernel_size=5, Q=-1.5)

        # 检查白点是否被抑制
        assert np.sum(filtered == 255) < np.sum(salt_noisy == 255)

    def test_contraharmonic_Q_range(self, filters, test_image):
        """测试不同Q值"""
        Q_values = [-2, -1, 0, 1, 2]

        for Q in Q_values:
            result = filters.contraharmonic_mean_filter(test_image, kernel_size=3, Q=Q)
            assert result.shape == test_image.shape
            assert not np.any(np.isnan(result))

    # ==================== 滤波器比较测试 ====================

    def test_filters_ordering_on_uniform(self, filters):
        """测试在均匀图像上的滤波器关系"""
        uniform = np.ones((50, 50), dtype=np.uint8) * 128

        arith = filters.arithmetic_mean_filter(uniform, kernel_size=3)
        geom = filters.geometric_mean_filter(uniform, kernel_size=3)
        harm = filters.harmonic_mean_filter(uniform, kernel_size=3)

        # 在均匀图像上，所有滤波器应该产生相似结果
        assert np.allclose(arith, uniform, atol=2)
        assert np.allclose(geom, uniform, atol=2)
        assert np.allclose(harm, uniform, atol=2)

    def test_filters_on_bright_image(self, filters):
        """测试在亮图像上的滤波器表现"""
        bright = np.ones((50, 50), dtype=np.uint8) * 200

        arith = filters.arithmetic_mean_filter(bright, kernel_size=3)
        geom = filters.geometric_mean_filter(bright, kernel_size=3)
        harm = filters.harmonic_mean_filter(bright, kernel_size=3)

        # 所有滤波器都应该保持亮度
        assert np.mean(arith) > 190
        assert np.mean(geom) > 190
        assert np.mean(harm) > 190

    def test_filters_on_dark_image(self, filters):
        """测试在暗图像上的滤波器表现"""
        dark = np.ones((50, 50), dtype=np.uint8) * 50

        arith = filters.arithmetic_mean_filter(dark, kernel_size=3)
        geom = filters.geometric_mean_filter(dark, kernel_size=3)
        harm = filters.harmonic_mean_filter(dark, kernel_size=3)

        # 所有滤波器都应该保持暗度
        assert np.mean(arith) < 60
        assert np.mean(geom) < 60
        assert np.mean(harm) < 60

    # ==================== 边界情况测试 ====================

    def test_all_black_image(self, filters):
        """测试全黑图像"""
        black = np.zeros((50, 50), dtype=np.uint8)

        arith = filters.arithmetic_mean_filter(black, kernel_size=3)
        geom = filters.geometric_mean_filter(black, kernel_size=3)
        harm = filters.harmonic_mean_filter(black, kernel_size=3)

        assert np.all(arith == 0)
        assert np.mean(geom) < 5  # 几何均值处理0值时可能有小偏差
        assert np.mean(harm) < 5

    def test_all_white_image(self, filters):
        """测试全白图像"""
        white = np.ones((50, 50), dtype=np.uint8) * 255

        arith = filters.arithmetic_mean_filter(white, kernel_size=3)
        geom = filters.geometric_mean_filter(white, kernel_size=3)
        harm = filters.harmonic_mean_filter(white, kernel_size=3)
        contra = filters.contraharmonic_mean_filter(white, kernel_size=3, Q=1.0)

        assert np.all(arith == 255)
        assert np.mean(geom) > 250
        assert np.mean(harm) > 250
        assert np.mean(contra) > 250

    def test_single_pixel_noise(self, filters):
        """测试单个噪声像素"""
        image = np.ones((50, 50), dtype=np.uint8) * 128
        image[25, 25] = 255  # 单个白点

        arith = filters.arithmetic_mean_filter(image, kernel_size=3)
        contra = filters.contraharmonic_mean_filter(image, kernel_size=3, Q=-1.0)

        # 噪声应该被平滑
        assert arith[25, 25] < 255
        assert contra[25, 25] < 255

    # ==================== 图像加载测试 ====================

    def test_load_image_success(self, filters, test_image, tmp_path):
        """测试成功加载图像"""
        temp_file = tmp_path / "test.png"
        cv2.imwrite(str(temp_file), test_image)

        success = filters.load_image(str(temp_file))
        assert success
        assert filters.image is not None

    def test_load_image_failure(self, filters):
        """测试加载不存在的图像"""
        success = filters.load_image("nonexistent.png")
        assert not success


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
