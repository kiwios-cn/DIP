"""
彩色图像处理测试
"""

import pytest
import numpy as np
import cv2
from color_image_processing import ColorImageProcessing


class TestColorImageProcessing:
    """测试彩色图像处理类"""

    @pytest.fixture
    def processor(self):
        """创建处理器对象"""
        return ColorImageProcessing()

    @pytest.fixture
    def test_image(self):
        """创建测试彩色图像"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(image, (50, 50), 30, (0, 0, 255), -1)  # 红色
        cv2.rectangle(image, (20, 20), (40, 40), (0, 255, 0), -1)  # 绿色
        return image

    # ==================== 高斯平滑测试 ====================

    def test_gaussian_smooth_rgb(self, processor, test_image):
        """测试RGB空间高斯平滑"""
        result = processor.gaussian_smooth(test_image, kernel_size=5, sigma=1.0, color_space='RGB')

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8
        assert len(result.shape) == 3  # 彩色图像

    def test_gaussian_smooth_hsv(self, processor, test_image):
        """测试HSV空间高斯平滑"""
        result = processor.gaussian_smooth(test_image, kernel_size=5, sigma=1.0, color_space='HSV')

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_gaussian_smooth_lab(self, processor, test_image):
        """测试LAB空间高斯平滑"""
        result = processor.gaussian_smooth(test_image, kernel_size=5, sigma=1.0, color_space='LAB')

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_gaussian_smooth_even_kernel(self, processor, test_image):
        """测试偶数核应该抛出异常"""
        with pytest.raises(ValueError):
            processor.gaussian_smooth(test_image, kernel_size=4)

    def test_gaussian_smooth_invalid_colorspace(self, processor, test_image):
        """测试无效颜色空间应该抛出异常"""
        with pytest.raises(ValueError):
            processor.gaussian_smooth(test_image, kernel_size=5, color_space='INVALID')

    # ==================== 双边滤波测试 ====================

    def test_bilateral_filter_basic(self, processor, test_image):
        """测试双边滤波基本功能"""
        result = processor.bilateral_filter(test_image, d=9, sigma_color=75, sigma_space=75)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_bilateral_filter_edge_preservation(self, processor):
        """测试双边滤波保边特性"""
        # 创建有明显边缘的图像
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :50] = [255, 0, 0]  # 左半边红色
        image[:, 50:] = [0, 0, 255]  # 右半边蓝色

        result = processor.bilateral_filter(image, d=9)

        # 检查边缘附近的像素，应该保持较大差异
        left_color = result[50, 45]
        right_color = result[50, 55]
        diff = np.abs(left_color.astype(float) - right_color.astype(float))

        # 边缘两侧颜色差异应该还比较大
        assert np.mean(diff) > 50

    # ==================== 均值平滑测试 ====================

    def test_mean_smooth_basic(self, processor, test_image):
        """测试均值平滑基本功能"""
        result = processor.mean_smooth(test_image, kernel_size=5, color_space='RGB')

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_mean_smooth_hsv(self, processor, test_image):
        """测试HSV空间均值平滑"""
        result = processor.mean_smooth(test_image, kernel_size=5, color_space='HSV')

        assert result.shape == test_image.shape

    # ==================== 中值平滑测试 ====================

    def test_median_smooth_basic(self, processor, test_image):
        """测试中值平滑基本功能"""
        result = processor.median_smooth(test_image, kernel_size=5)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    # ==================== 拉普拉斯锐化测试 ====================

    def test_laplacian_sharpen_rgb(self, processor, test_image):
        """测试RGB空间拉普拉斯锐化"""
        result = processor.laplacian_sharpen(test_image, kernel_type='4', alpha=1.0, color_space='RGB')

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_laplacian_sharpen_hsv(self, processor, test_image):
        """测试HSV空间拉普拉斯锐化"""
        result = processor.laplacian_sharpen(test_image, kernel_type='4', alpha=1.0, color_space='HSV')

        assert result.shape == test_image.shape

    def test_laplacian_sharpen_lab(self, processor, test_image):
        """测试LAB空间拉普拉斯锐化"""
        result = processor.laplacian_sharpen(test_image, kernel_type='4', alpha=1.0, color_space='LAB')

        assert result.shape == test_image.shape

    def test_laplacian_sharpen_kernel_types(self, processor, test_image):
        """测试不同拉普拉斯核类型"""
        for kernel_type in ['4', '8', 'enhanced']:
            result = processor.laplacian_sharpen(test_image, kernel_type=kernel_type)
            assert result.shape == test_image.shape

    def test_laplacian_sharpen_invalid_kernel(self, processor, test_image):
        """测试无效核类型应该抛出异常"""
        with pytest.raises(ValueError):
            processor.laplacian_sharpen(test_image, kernel_type='invalid')

    # ==================== Unsharp Masking测试 ====================

    def test_unsharp_masking_rgb(self, processor, test_image):
        """测试Unsharp Masking RGB"""
        result = processor.unsharp_masking(test_image, amount=1.5, color_space='RGB')

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_unsharp_masking_hsv(self, processor, test_image):
        """测试Unsharp Masking HSV"""
        result = processor.unsharp_masking(test_image, amount=1.5, color_space='HSV')

        assert result.shape == test_image.shape

    def test_unsharp_masking_lab(self, processor, test_image):
        """测试Unsharp Masking LAB"""
        result = processor.unsharp_masking(test_image, amount=1.5, color_space='LAB')

        assert result.shape == test_image.shape

    def test_unsharp_masking_with_threshold(self, processor, test_image):
        """测试带阈值的Unsharp Masking"""
        result = processor.unsharp_masking(test_image, amount=1.5, threshold=10)

        assert result.shape == test_image.shape

    def test_unsharp_masking_even_kernel(self, processor, test_image):
        """测试偶数核应该抛出异常"""
        with pytest.raises(ValueError):
            processor.unsharp_masking(test_image, kernel_size=4)

    # ==================== 高提升滤波测试 ====================

    def test_highboost_filter_basic(self, processor, test_image):
        """测试高提升滤波基本功能"""
        result = processor.highboost_filter(test_image, A=1.5, color_space='RGB')

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_highboost_filter_hsv(self, processor, test_image):
        """测试HSV空间高提升滤波"""
        result = processor.highboost_filter(test_image, A=1.5, color_space='HSV')

        assert result.shape == test_image.shape

    def test_highboost_filter_invalid_A(self, processor, test_image):
        """测试A<=1应该抛出异常"""
        with pytest.raises(ValueError):
            processor.highboost_filter(test_image, A=0.5)

    # ==================== 颜色空间效果测试 ====================

    def test_color_preservation(self, processor):
        """测试颜色保持"""
        # 创建纯色图像
        red_image = np.zeros((50, 50, 3), dtype=np.uint8)
        red_image[:, :] = [0, 0, 255]  # 纯红色

        # RGB平滑
        result_rgb = processor.gaussian_smooth(red_image, kernel_size=3, color_space='RGB')

        # HSV平滑（只平滑V通道，颜色应该保持）
        result_hsv = processor.gaussian_smooth(red_image, kernel_size=3, color_space='HSV')

        # HSV处理后红色应该保持得更好
        assert np.mean(result_hsv[:, :, 2]) > np.mean(result_rgb[:, :, 2]) * 0.9

    # ==================== 边界情况测试 ====================

    def test_all_black_image(self, processor):
        """测试全黑图像"""
        black = np.zeros((50, 50, 3), dtype=np.uint8)

        smooth = processor.gaussian_smooth(black, kernel_size=3)
        sharp = processor.laplacian_sharpen(black, alpha=1.0)

        assert np.all(smooth == 0)
        assert np.all(sharp == 0)

    def test_all_white_image(self, processor):
        """测试全白图像"""
        white = np.ones((50, 50, 3), dtype=np.uint8) * 255

        smooth = processor.gaussian_smooth(white, kernel_size=3)
        sharp = processor.laplacian_sharpen(white, alpha=1.0)

        assert np.all(smooth == 255)
        assert np.all(sharp == 255)

    def test_single_channel_vs_three_channel(self, processor):
        """测试单通道和三通道处理的一致性"""
        # 创建灰度图像
        gray = np.ones((50, 50), dtype=np.uint8) * 128

        # 转换为三通道
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 平滑处理
        result = processor.gaussian_smooth(color, kernel_size=3)

        # 三个通道应该相同或非常接近
        assert np.allclose(result[:, :, 0], result[:, :, 1], atol=2)
        assert np.allclose(result[:, :, 1], result[:, :, 2], atol=2)

    # ==================== 图像加载测试 ====================

    def test_load_image_success(self, processor, test_image, tmp_path):
        """测试成功加载图像"""
        temp_file = tmp_path / "test.png"
        cv2.imwrite(str(temp_file), test_image)

        success = processor.load_image(str(temp_file))
        assert success
        assert processor.image is not None
        assert len(processor.image.shape) == 3

    def test_load_image_failure(self, processor):
        """测试加载不存在的图像"""
        success = processor.load_image("nonexistent.png")
        assert not success

    # ==================== 参数范围测试 ====================

    def test_alpha_range(self, processor, test_image):
        """测试不同alpha值"""
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            result = processor.laplacian_sharpen(test_image, alpha=alpha)
            assert result.shape == test_image.shape

    def test_amount_range(self, processor, test_image):
        """测试不同amount值"""
        for amount in [0.5, 1.0, 1.5, 2.0, 2.5]:
            result = processor.unsharp_masking(test_image, amount=amount)
            assert result.shape == test_image.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
