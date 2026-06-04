"""
统计排序滤波器测试
"""

import pytest
import numpy as np
import cv2
from order_statistic_filters import OrderStatisticFilters


class TestOrderStatisticFilters:
    """测试统计排序滤波器类"""

    @pytest.fixture
    def filters(self):
        """创建滤波器对象"""
        return OrderStatisticFilters()

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (50, 50), 30, 128, -1)
        return image

    @pytest.fixture
    def sp_noisy_image(self, test_image):
        """创建椒盐噪声图像"""
        noisy = test_image.copy()
        # 添加盐噪声
        num_salt = 100
        coords = [np.random.randint(0, i, num_salt) for i in test_image.shape]
        noisy[coords[0], coords[1]] = 255
        # 添加椒噪声
        num_pepper = 100
        coords = [np.random.randint(0, i, num_pepper) for i in test_image.shape]
        noisy[coords[0], coords[1]] = 0
        return noisy

    # ==================== 中值滤波器测试 ====================

    def test_median_filter_basic(self, filters, test_image):
        """测试中值滤波器基本功能"""
        result = filters.median_filter(test_image, kernel_size=3)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_median_filter_kernel_sizes(self, filters, test_image):
        """测试不同卷积核大小"""
        for size in [3, 5, 7]:
            result = filters.median_filter(test_image, kernel_size=size)
            assert result.shape == test_image.shape

    def test_median_filter_even_kernel(self, filters, test_image):
        """测试偶数卷积核应该抛出异常"""
        with pytest.raises(ValueError):
            filters.median_filter(test_image, kernel_size=4)

    def test_median_filter_salt_pepper(self, filters):
        """测试中值滤波去除椒盐噪声"""
        # 使用亮度较高的测试图像
        test_img = np.ones((100, 100), dtype=np.uint8) * 128
        cv2.circle(test_img, (50, 50), 30, 200, -1)

        # 添加椒盐噪声
        noisy = test_img.copy()
        # 盐噪声
        num_salt = 100
        coords = [np.random.randint(0, i, num_salt) for i in test_img.shape]
        noisy[coords[0], coords[1]] = 255
        # 椒噪声
        num_pepper = 100
        coords = [np.random.randint(0, i, num_pepper) for i in test_img.shape]
        noisy[coords[0], coords[1]] = 0

        filtered = filters.median_filter(noisy, kernel_size=5)

        # 检查噪声是否被去除
        assert np.sum(filtered == 0) < np.sum(noisy == 0)
        assert np.sum(filtered == 255) < np.sum(noisy == 255)

    # ==================== 最大值滤波器测试 ====================

    def test_max_filter_basic(self, filters, test_image):
        """测试最大值滤波器基本功能"""
        result = filters.max_filter(test_image, kernel_size=3)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_max_filter_kernel_sizes(self, filters, test_image):
        """测试不同卷积核大小"""
        for size in [3, 5, 7]:
            result = filters.max_filter(test_image, kernel_size=size)
            assert result.shape == test_image.shape

    def test_max_filter_even_kernel(self, filters, test_image):
        """测试偶数卷积核应该抛出异常"""
        with pytest.raises(ValueError):
            filters.max_filter(test_image, kernel_size=4)

    def test_max_filter_removes_pepper(self, filters, test_image):
        """测试最大值滤波器去除椒噪声"""
        # 添加椒噪声（黑点）
        pepper_noisy = test_image.copy()
        num_pepper = 200
        coords = [np.random.randint(0, i, num_pepper) for i in test_image.shape]
        pepper_noisy[coords[0], coords[1]] = 0

        filtered = filters.max_filter(pepper_noisy, kernel_size=5)

        # 检查黑点是否被减少
        assert np.sum(filtered == 0) < np.sum(pepper_noisy == 0)

    def test_max_filter_brightness(self, filters, test_image):
        """测试最大值滤波器增加亮度"""
        filtered = filters.max_filter(test_image, kernel_size=3)

        # 平均亮度应该增加
        assert np.mean(filtered) >= np.mean(test_image)

    # ==================== 最小值滤波器测试 ====================

    def test_min_filter_basic(self, filters, test_image):
        """测试最小值滤波器基本功能"""
        result = filters.min_filter(test_image, kernel_size=3)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_min_filter_kernel_sizes(self, filters, test_image):
        """测试不同卷积核大小"""
        for size in [3, 5, 7]:
            result = filters.min_filter(test_image, kernel_size=size)
            assert result.shape == test_image.shape

    def test_min_filter_even_kernel(self, filters, test_image):
        """测试偶数卷积核应该抛出异常"""
        with pytest.raises(ValueError):
            filters.min_filter(test_image, kernel_size=4)

    def test_min_filter_removes_salt(self, filters, test_image):
        """测试最小值滤波器去除盐噪声"""
        # 添加盐噪声（白点）
        salt_noisy = test_image.copy()
        num_salt = 200
        coords = [np.random.randint(0, i, num_salt) for i in test_image.shape]
        salt_noisy[coords[0], coords[1]] = 255

        filtered = filters.min_filter(salt_noisy, kernel_size=5)

        # 检查白点是否被减少
        assert np.sum(filtered == 255) < np.sum(salt_noisy == 255)

    def test_min_filter_darkness(self, filters, test_image):
        """测试最小值滤波器降低亮度"""
        filtered = filters.min_filter(test_image, kernel_size=3)

        # 平均亮度应该降低
        assert np.mean(filtered) <= np.mean(test_image)

    # ==================== 中点滤波器测试 ====================

    def test_midpoint_filter_basic(self, filters, test_image):
        """测试中点滤波器基本功能"""
        result = filters.midpoint_filter(test_image, kernel_size=3)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_midpoint_filter_kernel_sizes(self, filters, test_image):
        """测试不同卷积核大小"""
        for size in [3, 5, 7]:
            result = filters.midpoint_filter(test_image, kernel_size=size)
            assert result.shape == test_image.shape

    def test_midpoint_filter_even_kernel(self, filters, test_image):
        """测试偶数卷积核应该抛出异常"""
        with pytest.raises(ValueError):
            filters.midpoint_filter(test_image, kernel_size=4)

    def test_midpoint_relationship(self, filters):
        """测试中点滤波器与最大最小值的关系"""
        image = np.ones((50, 50), dtype=np.uint8) * 128

        max_img = filters.max_filter(image, kernel_size=3)
        min_img = filters.min_filter(image, kernel_size=3)
        midpoint = filters.midpoint_filter(image, kernel_size=3)

        # 中点应该在最大值和最小值之间
        assert np.all(midpoint >= min_img)
        assert np.all(midpoint <= max_img)

    # ==================== 修正阿尔法均值滤波器测试 ====================

    def test_alpha_trimmed_mean_filter_basic(self, filters, test_image):
        """测试修正阿尔法均值滤波器基本功能"""
        result = filters.alpha_trimmed_mean_filter(test_image, kernel_size=3, d=2)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_alpha_trimmed_mean_filter_kernel_sizes(self, filters, test_image):
        """测试不同卷积核大小"""
        for size in [3, 5, 7]:
            result = filters.alpha_trimmed_mean_filter(test_image, kernel_size=size, d=2)
            assert result.shape == test_image.shape

    def test_alpha_trimmed_mean_filter_even_kernel(self, filters, test_image):
        """测试偶数卷积核应该抛出异常"""
        with pytest.raises(ValueError):
            filters.alpha_trimmed_mean_filter(test_image, kernel_size=4, d=2)

    def test_alpha_trimmed_mean_filter_odd_d(self, filters, test_image):
        """测试奇数d应该抛出异常"""
        with pytest.raises(ValueError):
            filters.alpha_trimmed_mean_filter(test_image, kernel_size=3, d=3)

    def test_alpha_trimmed_mean_filter_d_too_large(self, filters, test_image):
        """测试d过大应该抛出异常"""
        with pytest.raises(ValueError):
            filters.alpha_trimmed_mean_filter(test_image, kernel_size=3, d=10)

    def test_alpha_trimmed_mean_d_zero(self, filters, test_image):
        """测试d=0时接近算术均值"""
        alpha = filters.alpha_trimmed_mean_filter(test_image, kernel_size=3, d=0)
        arith = cv2.blur(test_image, (3, 3))

        # 结果应该非常接近
        diff = np.abs(alpha.astype(float) - arith.astype(float))
        assert np.mean(diff) < 1.0

    def test_alpha_trimmed_mean_d_values(self, filters, test_image):
        """测试不同d值"""
        d_values = [0, 2, 4, 6]

        for d in d_values:
            result = filters.alpha_trimmed_mean_filter(test_image, kernel_size=5, d=d)
            assert result.shape == test_image.shape
            assert not np.any(np.isnan(result))

    def test_alpha_trimmed_mean_mixed_noise(self, filters, sp_noisy_image):
        """测试修正阿尔法均值滤波器对混合噪声的效果"""
        filtered = filters.alpha_trimmed_mean_filter(sp_noisy_image, kernel_size=5, d=4)

        # 应该减少噪声
        assert np.sum(filtered == 0) < np.sum(sp_noisy_image == 0)
        assert np.sum(filtered == 255) < np.sum(sp_noisy_image == 255)

    # ==================== 滤波器比较测试 ====================

    def test_filters_on_uniform_image(self, filters):
        """测试在均匀图像上的滤波器表现"""
        uniform = np.ones((50, 50), dtype=np.uint8) * 128

        median = filters.median_filter(uniform, kernel_size=3)
        max_f = filters.max_filter(uniform, kernel_size=3)
        min_f = filters.min_filter(uniform, kernel_size=3)
        midpoint = filters.midpoint_filter(uniform, kernel_size=3)
        alpha = filters.alpha_trimmed_mean_filter(uniform, kernel_size=3, d=2)

        # 在均匀图像上，所有滤波器应该产生相似结果
        assert np.allclose(median, uniform, atol=2)
        assert np.allclose(max_f, uniform, atol=2)
        assert np.allclose(min_f, uniform, atol=2)
        assert np.allclose(midpoint, uniform, atol=2)
        assert np.allclose(alpha, uniform, atol=2)

    def test_max_min_relationship(self, filters, test_image):
        """测试最大值和最小值滤波器的关系"""
        max_img = filters.max_filter(test_image, kernel_size=3)
        min_img = filters.min_filter(test_image, kernel_size=3)

        # 最大值应该大于等于最小值
        assert np.all(max_img >= min_img)

    # ==================== 边界情况测试 ====================

    def test_all_black_image(self, filters):
        """测试全黑图像"""
        black = np.zeros((50, 50), dtype=np.uint8)

        median = filters.median_filter(black, kernel_size=3)
        max_f = filters.max_filter(black, kernel_size=3)
        min_f = filters.min_filter(black, kernel_size=3)
        midpoint = filters.midpoint_filter(black, kernel_size=3)
        alpha = filters.alpha_trimmed_mean_filter(black, kernel_size=3, d=2)

        assert np.all(median == 0)
        assert np.all(max_f == 0)
        assert np.all(min_f == 0)
        assert np.all(midpoint == 0)
        assert np.all(alpha == 0)

    def test_all_white_image(self, filters):
        """测试全白图像"""
        white = np.ones((50, 50), dtype=np.uint8) * 255

        median = filters.median_filter(white, kernel_size=3)
        max_f = filters.max_filter(white, kernel_size=3)
        min_f = filters.min_filter(white, kernel_size=3)
        midpoint = filters.midpoint_filter(white, kernel_size=3)
        alpha = filters.alpha_trimmed_mean_filter(white, kernel_size=3, d=2)

        assert np.all(median == 255)
        assert np.all(max_f == 255)
        assert np.all(min_f == 255)
        assert np.all(midpoint == 255)
        assert np.all(alpha == 255)

    def test_single_pixel_noise(self, filters):
        """测试单个噪声像素"""
        image = np.ones((50, 50), dtype=np.uint8) * 128
        image[25, 25] = 255  # 单个白点

        median = filters.median_filter(image, kernel_size=3)

        # 噪声应该被去除
        assert median[25, 25] < 255
        assert median[25, 25] >= 128

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
