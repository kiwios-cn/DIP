"""
空间域滤波测试用例
测试均值滤波、中值滤波、拉普拉斯滤波的正确性
"""

import pytest
import numpy as np
import cv2
from spatial_filtering import SpatialFiltering


class TestSpatialFiltering:
    """测试空间域滤波类"""

    @pytest.fixture
    def filter_obj(self):
        """创建滤波器实例"""
        return SpatialFiltering()

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        # 创建简单的测试图像
        image = np.array([[100, 100, 100, 100, 100],
                         [100, 150, 150, 150, 100],
                         [100, 150, 200, 150, 100],
                         [100, 150, 150, 150, 100],
                         [100, 100, 100, 100, 100]], dtype=np.uint8)
        return image

    @pytest.fixture
    def noisy_image(self):
        """创建带噪声的测试图像"""
        image = np.ones((5, 5), dtype=np.uint8) * 128
        # 添加椒盐噪声
        image[1, 1] = 0    # 椒噪声
        image[3, 3] = 255  # 盐噪声
        return image

    # ==================== 均值滤波测试 ====================

    def test_mean_filter_basic(self, filter_obj, test_image):
        """测试基本均值滤波"""
        result = filter_obj.mean_filter(test_image, kernel_size=3)

        # 验证结果形状
        assert result.shape == test_image.shape

        # 验证数据类型
        assert result.dtype == np.uint8

        # 验证平滑效果（中心像素应该接近周围像素的平均值）
        center = result[2, 2]
        assert 100 < center < 200

    def test_mean_filter_kernel_sizes(self, filter_obj, test_image):
        """测试不同卷积核大小"""
        result_3x3 = filter_obj.mean_filter(test_image, 3)
        result_5x5 = filter_obj.mean_filter(test_image, 5)

        # 更大的卷积核应该产生更平滑的结果
        assert result_3x3.shape == result_5x5.shape

    def test_weighted_mean_filter(self, filter_obj, test_image):
        """测试加权均值滤波"""
        result = filter_obj.weighted_mean_filter(test_image, kernel_size=3)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_custom_mean_filter(self, filter_obj, test_image):
        """测试自定义卷积核"""
        # 创建简单的3x3均值核
        kernel = np.ones((3, 3), dtype=np.float32)
        result = filter_obj.custom_mean_filter(test_image, kernel)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    # ==================== 中值滤波测试 ====================

    def test_median_filter_basic(self, filter_obj, noisy_image):
        """测试基本中值滤波"""
        result = filter_obj.median_filter(noisy_image, kernel_size=3)

        # 验证结果形状
        assert result.shape == noisy_image.shape

        # 验证椒盐噪声被去除
        # 原图中(1,1)是0，中值滤波后应该接近128
        assert result[1, 1] > 50

        # 原图中(3,3)是255，中值滤波后应该接近128
        assert result[3, 3] < 200

    def test_median_filter_preserves_edges(self, filter_obj):
        """测试中值滤波保持边缘"""
        # 创建阶跃边缘
        image = np.zeros((10, 10), dtype=np.uint8)
        image[:, 5:] = 255

        result = filter_obj.median_filter(image, kernel_size=3)

        # 边缘应该被保持（虽然可能略有模糊）
        assert np.any(result[:, 4] < 128)  # 左侧暗
        assert np.any(result[:, 6] > 128)  # 右侧亮

    def test_median_filter_kernel_sizes(self, filter_obj, noisy_image):
        """测试不同卷积核大小"""
        result_3x3 = filter_obj.median_filter(noisy_image, 3)
        result_5x5 = filter_obj.median_filter(noisy_image, 5)

        assert result_3x3.shape == result_5x5.shape

    def test_adaptive_median_filter(self, filter_obj, noisy_image):
        """测试自适应中值滤波"""
        result = filter_obj.adaptive_median_filter(noisy_image, max_kernel_size=7)

        assert result.shape == noisy_image.shape
        assert result.dtype == np.uint8

        # 验证噪声被去除
        assert result[1, 1] > 50
        assert result[3, 3] < 200

    # ==================== 拉普拉斯滤波测试 ====================

    def test_laplacian_filter_4_neighbor(self, filter_obj, test_image):
        """测试4邻域拉普拉斯"""
        result = filter_obj.laplacian_filter(test_image, kernel_type='4')

        # 验证结果形状
        assert result.shape == test_image.shape

        # 拉普拉斯响应可能有负值
        assert result.dtype == np.float32

    def test_laplacian_filter_8_neighbor(self, filter_obj, test_image):
        """测试8邻域拉普拉斯"""
        result = filter_obj.laplacian_filter(test_image, kernel_type='8')

        assert result.shape == test_image.shape
        assert result.dtype == np.float32

    def test_laplacian_filter_enhanced(self, filter_obj, test_image):
        """测试增强型拉普拉斯"""
        result = filter_obj.laplacian_filter(test_image, kernel_type='enhanced')

        assert result.shape == test_image.shape
        assert result.dtype == np.float32

    def test_laplacian_filter_invalid_type(self, filter_obj, test_image):
        """测试无效的卷积核类型"""
        with pytest.raises(ValueError):
            filter_obj.laplacian_filter(test_image, kernel_type='invalid')

    def test_laplacian_sharpen(self, filter_obj, test_image):
        """测试拉普拉斯锐化"""
        result = filter_obj.laplacian_sharpen(test_image, kernel_type='4', alpha=1.0)

        # 验证结果
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

        # 锐化后的图像应该有更高的对比度
        # 但这个测试比较主观，只验证基本属性

    def test_laplacian_sharpen_alpha(self, filter_obj, test_image):
        """测试不同锐化强度"""
        result_weak = filter_obj.laplacian_sharpen(test_image, alpha=0.5)
        result_strong = filter_obj.laplacian_sharpen(test_image, alpha=2.0)

        assert result_weak.shape == result_strong.shape

    def test_log_filter(self, filter_obj, test_image):
        """测试LoG滤波器"""
        result = filter_obj.log_filter(test_image, sigma=1.0)

        assert result.shape == test_image.shape
        assert result.dtype == np.float32

    def test_log_filter_sigma(self, filter_obj, test_image):
        """测试不同sigma值"""
        result_small = filter_obj.log_filter(test_image, sigma=0.5)
        result_large = filter_obj.log_filter(test_image, sigma=2.0)

        assert result_small.shape == result_large.shape

    # ==================== 噪声添加测试 ====================

    def test_add_gaussian_noise(self, filter_obj, test_image):
        """测试添加高斯噪声"""
        noisy = filter_obj.add_noise(test_image, 'gaussian', sigma=25)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

        # 噪声图像应该与原图不同
        assert not np.array_equal(noisy, test_image)

    def test_add_salt_pepper_noise(self, filter_obj, test_image):
        """测试添加椒盐噪声"""
        noisy = filter_obj.add_noise(test_image, 'salt_pepper', prob=0.1)

        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

        # 应该有一些0和255的像素
        assert np.any(noisy == 0) or np.any(noisy == 255)

    # ==================== 图像加载测试 ====================

    def test_load_image_success(self, filter_obj, tmp_path):
        """测试成功加载图像"""
        # 创建临时测试图像
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), test_img)

        # 测试加载
        success = filter_obj.load_image(str(img_path))
        assert success
        assert filter_obj.image is not None
        assert filter_obj.gray_image is not None

    def test_load_image_failure(self, filter_obj):
        """测试加载不存在的图像"""
        success = filter_obj.load_image("nonexistent.png")
        assert not success

    def test_load_grayscale_image(self, filter_obj, tmp_path):
        """测试加载灰度图像"""
        test_img = np.ones((100, 100), dtype=np.uint8) * 128
        img_path = tmp_path / "test_gray.png"
        cv2.imwrite(str(img_path), test_img)

        success = filter_obj.load_image(str(img_path))
        assert success
        assert filter_obj.gray_image is not None

    # ==================== 边界情况测试 ====================

    def test_filter_on_small_image(self, filter_obj):
        """测试小图像滤波"""
        small_image = np.array([[100, 150],
                               [150, 200]], dtype=np.uint8)

        result = filter_obj.mean_filter(small_image, 3)
        assert result.shape == small_image.shape

    def test_filter_on_uniform_image(self, filter_obj):
        """测试均匀图像滤波"""
        uniform = np.ones((10, 10), dtype=np.uint8) * 128

        mean_result = filter_obj.mean_filter(uniform, 3)
        median_result = filter_obj.median_filter(uniform, 3)

        # 均匀图像滤波后应该保持不变
        np.testing.assert_array_almost_equal(mean_result, uniform, decimal=0)
        np.testing.assert_array_equal(median_result, uniform)

    def test_filter_on_zero_image(self, filter_obj):
        """测试全零图像"""
        zero_image = np.zeros((10, 10), dtype=np.uint8)

        result = filter_obj.mean_filter(zero_image, 3)
        assert np.all(result == 0)

    def test_filter_on_max_image(self, filter_obj):
        """测试全255图像"""
        max_image = np.ones((10, 10), dtype=np.uint8) * 255

        result = filter_obj.mean_filter(max_image, 3)
        assert np.all(result == 255)

    # ==================== 性能测试 ====================

    def test_filter_preserves_shape(self, filter_obj):
        """测试所有滤波器保持图像形状"""
        test_shapes = [(10, 10), (50, 50), (100, 100)]

        for shape in test_shapes:
            image = np.random.randint(0, 256, shape, dtype=np.uint8)

            mean_result = filter_obj.mean_filter(image, 3)
            median_result = filter_obj.median_filter(image, 3)
            laplacian_result = filter_obj.laplacian_filter(image, '4')

            assert mean_result.shape == shape
            assert median_result.shape == shape
            assert laplacian_result.shape == shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
