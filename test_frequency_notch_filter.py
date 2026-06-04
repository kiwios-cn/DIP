"""
频域陷波滤波器测试用例
测试理想、巴特沃斯、高斯陷波滤波器的正确性
"""

import pytest
import numpy as np
import cv2
from frequency_notch_filter import FrequencyNotchFilter


class TestFrequencyNotchFilter:
    """测试频域陷波滤波器类"""

    @pytest.fixture
    def filter_obj(self):
        """创建滤波器实例"""
        return FrequencyNotchFilter()

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        image = np.ones((128, 128), dtype=np.uint8) * 128
        cv2.circle(image, (64, 64), 30, 200, -1)
        return image

    @pytest.fixture
    def noisy_image(self, filter_obj, test_image):
        """创建带周期噪声的测试图像"""
        noisy = filter_obj.add_periodic_noise(
            test_image,
            frequencies=[(0, 20), (20, 0)],
            amplitudes=[30, 30]
        )
        return noisy

    # ==================== FFT变换测试 ====================

    def test_fft_transform(self, filter_obj, test_image):
        """测试FFT变换"""
        fft, fft_shifted = filter_obj.fft_transform(test_image)

        # 验证形状
        assert fft.shape == test_image.shape
        assert fft_shifted.shape == test_image.shape

        # 验证复数类型
        assert np.iscomplexobj(fft)
        assert np.iscomplexobj(fft_shifted)

    def test_ifft_transform(self, filter_obj, test_image):
        """测试逆FFT变换"""
        # 先FFT
        _, fft_shifted = filter_obj.fft_transform(test_image)

        # 再逆FFT
        reconstructed = filter_obj.ifft_transform(fft_shifted)

        # 验证形状
        assert reconstructed.shape == test_image.shape

        # 验证重建精度（允许小误差）
        np.testing.assert_allclose(reconstructed, test_image, atol=1e-5)

    def test_fft_energy_conservation(self, filter_obj, test_image):
        """测试FFT能量守恒（Parseval定理）"""
        # 空间域能量
        energy_spatial = np.sum(test_image.astype(np.float32)**2)

        # 频率域能量
        fft, _ = filter_obj.fft_transform(test_image)
        energy_freq = np.sum(np.abs(fft)**2) / test_image.size

        # 验证能量守恒（相对误差在1%以内）
        np.testing.assert_allclose(energy_spatial, energy_freq, rtol=0.01)

    # ==================== 陷波掩码测试 ====================

    def test_create_notch_mask_ideal(self, filter_obj):
        """测试理想陷波掩码"""
        shape = (100, 100)
        notch_positions = [(60, 60)]
        radius = 5

        mask = filter_obj.create_notch_mask(shape, notch_positions, radius, 'ideal')

        # 验证形状
        assert mask.shape == shape

        # 验证值域 [0, 1]
        assert np.all(mask >= 0) and np.all(mask <= 1)

        # 验证对称性：应该在(60,60)和(40,40)都有陷波
        center = 50
        # 检查(60,60)附近应该被抑制
        assert mask[60, 60] < 0.1
        # 检查对称点(40,40)附近也应该被抑制
        assert mask[40, 40] < 0.1

    def test_create_notch_mask_butterworth(self, filter_obj):
        """测试巴特沃斯陷波掩码"""
        shape = (100, 100)
        notch_positions = [(60, 60)]
        radius = 5

        mask = filter_obj.create_notch_mask(shape, notch_positions, radius, 'butterworth', order=2)

        # 验证形状
        assert mask.shape == shape

        # 验证值域 [0, 1]
        assert np.all(mask >= 0) and np.all(mask <= 1)

        # 验证平滑性：巴特沃斯应该比理想滤波器更平滑
        # 中心点应该被抑制但不是完全为0
        assert mask[60, 60] < 0.5

    def test_create_notch_mask_gaussian(self, filter_obj):
        """测试高斯陷波掩码"""
        shape = (100, 100)
        notch_positions = [(60, 60)]
        radius = 5

        mask = filter_obj.create_notch_mask(shape, notch_positions, radius, 'gaussian')

        # 验证形状
        assert mask.shape == shape

        # 验证值域 [0, 1]
        assert np.all(mask >= 0) and np.all(mask <= 1)

    def test_create_notch_mask_invalid_type(self, filter_obj):
        """测试无效的滤波器类型"""
        with pytest.raises(ValueError):
            filter_obj.create_notch_mask((100, 100), [(50, 50)], 5, 'invalid')

    def test_create_notch_mask_multiple_positions(self, filter_obj):
        """测试多个陷波位置"""
        shape = (100, 100)
        notch_positions = [(60, 60), (70, 70)]
        radius = 5

        mask = filter_obj.create_notch_mask(shape, notch_positions, radius, 'ideal')

        # 验证所有陷波点都被抑制
        assert mask[60, 60] < 0.1
        assert mask[40, 40] < 0.1  # 对称点
        assert mask[70, 70] < 0.1
        assert mask[30, 30] < 0.1  # 对称点

    # ==================== 陷波滤波器测试 ====================

    def test_ideal_notch_reject(self, filter_obj, noisy_image):
        """测试理想陷波抑制滤波器"""
        notch_positions = [(64, 84), (84, 64)]  # 基于噪声频率
        radius = 5

        result = filter_obj.ideal_notch_reject(noisy_image, notch_positions, radius)

        # 验证结果形状
        assert result.shape == noisy_image.shape

        # 验证数据类型
        assert result.dtype == np.uint8

        # 验证噪声被减弱（通过方差降低判断）
        # 噪声图像应该比滤波后的方差更大
        assert np.var(noisy_image) > np.var(result)

    def test_butterworth_notch_reject(self, filter_obj, noisy_image):
        """测试巴特沃斯陷波抑制滤波器"""
        notch_positions = [(64, 84), (84, 64)]
        radius = 5

        result = filter_obj.butterworth_notch_reject(noisy_image, notch_positions, radius, order=2)

        assert result.shape == noisy_image.shape
        assert result.dtype == np.uint8

    def test_gaussian_notch_reject(self, filter_obj, noisy_image):
        """测试高斯陷波抑制滤波器"""
        notch_positions = [(64, 84), (84, 64)]
        radius = 5

        result = filter_obj.gaussian_notch_reject(noisy_image, notch_positions, radius)

        assert result.shape == noisy_image.shape
        assert result.dtype == np.uint8

    def test_ideal_notch_pass(self, filter_obj, noisy_image):
        """测试理想陷波通过滤波器"""
        notch_positions = [(64, 84), (84, 64)]
        radius = 5

        result = filter_obj.ideal_notch_pass(noisy_image, notch_positions, radius)

        # 验证结果形状
        assert result.shape == noisy_image.shape
        assert result.dtype == np.uint8

        # 陷波通过应该只保留噪声部分
        # 结果的平均值应该接近0（只有高频噪声）
        assert np.mean(result) < np.mean(noisy_image)

    def test_notch_filter_different_orders(self, filter_obj, noisy_image):
        """测试不同阶数的巴特沃斯滤波器"""
        notch_positions = [(64, 84)]
        radius = 5

        result_order1 = filter_obj.butterworth_notch_reject(noisy_image, notch_positions, radius, order=1)
        result_order4 = filter_obj.butterworth_notch_reject(noisy_image, notch_positions, radius, order=4)

        # 高阶滤波器应该更接近理想滤波器
        assert result_order1.shape == result_order4.shape

    # ==================== 噪声添加测试 ====================

    def test_add_periodic_noise(self, filter_obj, test_image):
        """测试添加周期性噪声"""
        frequencies = [(0, 20), (20, 0)]
        amplitudes = [30, 30]

        noisy = filter_obj.add_periodic_noise(test_image, frequencies, amplitudes)

        # 验证形状和类型
        assert noisy.shape == test_image.shape
        assert noisy.dtype == np.uint8

        # 噪声图像应该与原图不同
        assert not np.array_equal(noisy, test_image)

        # 噪声应该增加图像方差
        assert np.var(noisy) > np.var(test_image)

    def test_add_periodic_noise_multiple_frequencies(self, filter_obj, test_image):
        """测试添加多个频率的噪声"""
        frequencies = [(10, 10), (20, 0), (0, 20)]
        amplitudes = [20, 25, 30]

        noisy = filter_obj.add_periodic_noise(test_image, frequencies, amplitudes)

        assert noisy.shape == test_image.shape
        # 应该能看到噪声影响
        assert np.std(noisy) > np.std(test_image)

    # ==================== 峰值检测测试 ====================

    def test_detect_peaks_in_spectrum(self, filter_obj, noisy_image):
        """测试频谱峰值检测"""
        _, fft_shifted = filter_obj.fft_transform(noisy_image)

        peaks = filter_obj.detect_peaks_in_spectrum(fft_shifted, threshold_percentile=99.5)

        # 应该检测到峰值
        assert len(peaks) > 0

        # 峰值应该是元组
        for peak in peaks:
            assert isinstance(peak, tuple)
            assert len(peak) == 2

    def test_detect_peaks_excludes_dc(self, filter_obj, test_image):
        """测试峰值检测排除DC分量"""
        _, fft_shifted = filter_obj.fft_transform(test_image)

        peaks = filter_obj.detect_peaks_in_spectrum(fft_shifted, threshold_percentile=99.9)

        # 中心点（DC分量）不应该在峰值列表中
        H, W = test_image.shape
        center = (H // 2, W // 2)

        assert center not in peaks

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

    # ==================== 边界情况测试 ====================

    def test_notch_filter_uniform_image(self, filter_obj):
        """测试均匀图像的陷波滤波"""
        uniform = np.ones((100, 100), dtype=np.uint8) * 128
        notch_positions = [(60, 60)]
        radius = 5

        result = filter_obj.ideal_notch_reject(uniform, notch_positions, radius)

        # 均匀图像滤波后应该保持均匀（无高频分量）
        assert result.shape == uniform.shape
        # 允许一些数值误差
        np.testing.assert_allclose(result, uniform, atol=1)

    def test_notch_filter_zero_image(self, filter_obj):
        """测试全零图像"""
        zero_image = np.zeros((100, 100), dtype=np.uint8)
        notch_positions = [(60, 60)]
        radius = 5

        result = filter_obj.ideal_notch_reject(zero_image, notch_positions, radius)

        # 零图像滤波后应该仍是零
        np.testing.assert_allclose(result, zero_image, atol=1)

    def test_notch_filter_small_radius(self, filter_obj, test_image):
        """测试小陷波半径"""
        notch_positions = [(64, 64)]
        radius = 1

        result = filter_obj.ideal_notch_reject(test_image, notch_positions, radius)

        # 应该能正常工作
        assert result.shape == test_image.shape

    def test_notch_filter_large_radius(self, filter_obj, test_image):
        """测试大陷波半径"""
        notch_positions = [(64, 64)]
        radius = 30

        result = filter_obj.ideal_notch_reject(test_image, notch_positions, radius)

        # 应该能正常工作，但会移除更多频率分量
        assert result.shape == test_image.shape

    def test_notch_at_edge(self, filter_obj, test_image):
        """测试边缘位置的陷波"""
        H, W = test_image.shape
        # 边缘位置
        notch_positions = [(5, 5), (H-5, W-5)]
        radius = 3

        result = filter_obj.ideal_notch_reject(test_image, notch_positions, radius)

        assert result.shape == test_image.shape

    # ==================== 对称性测试 ====================

    def test_notch_symmetry(self, filter_obj):
        """测试陷波点的对称性"""
        shape = (100, 100)
        notch_positions = [(70, 60)]  # 只指定一个点
        radius = 5

        mask = filter_obj.create_notch_mask(shape, notch_positions, radius, 'ideal')

        # 验证对称点也被抑制
        # 原点: (70, 60), 中心: (50, 50)
        # 对称点: (50 - (70-50), 50 - (60-50)) = (30, 40)
        assert mask[70, 60] < 0.1
        assert mask[30, 40] < 0.1

    # ==================== 滤波器类型比较测试 ====================

    def test_filter_types_comparison(self, filter_obj, noisy_image):
        """测试不同滤波器类型的效果"""
        notch_positions = [(64, 84)]
        radius = 5

        ideal = filter_obj.ideal_notch_reject(noisy_image, notch_positions, radius)
        butterworth = filter_obj.butterworth_notch_reject(noisy_image, notch_positions, radius, order=2)
        gaussian = filter_obj.gaussian_notch_reject(noisy_image, notch_positions, radius)

        # 所有滤波器应该都能去除噪声
        assert np.var(ideal) < np.var(noisy_image)
        assert np.var(butterworth) < np.var(noisy_image)
        assert np.var(gaussian) < np.var(noisy_image)

        # 理想滤波器通常去噪效果最强（但可能有振铃）
        # 这里只验证都有效果，不比较强度

    def test_notch_reject_vs_pass(self, filter_obj, test_image):
        """测试陷波抑制和陷波通过的互补性"""
        notch_positions = [(64, 84)]
        radius = 5

        # 添加噪声
        noisy = filter_obj.add_periodic_noise(test_image, [(0, 20)], [30])

        # 陷波抑制（去除噪声）
        reject_result = filter_obj.ideal_notch_reject(noisy, notch_positions, radius)

        # 陷波通过（只保留噪声）
        pass_result = filter_obj.ideal_notch_pass(noisy, notch_positions, radius)

        # 两者应该互补（虽然不是完全的线性叠加）
        # 验证pass_result主要包含噪声（平均值应该小）
        assert np.mean(pass_result) < np.mean(noisy)

    # ==================== 性能测试 ====================

    def test_filter_preserves_shape(self, filter_obj):
        """测试所有滤波器保持图像形状"""
        test_shapes = [(64, 64), (128, 128), (256, 256)]
        notch_positions = [(40, 40)]
        radius = 5

        for shape in test_shapes:
            image = np.random.randint(0, 256, shape, dtype=np.uint8)

            ideal = filter_obj.ideal_notch_reject(image, notch_positions, radius)
            butterworth = filter_obj.butterworth_notch_reject(image, notch_positions, radius)
            gaussian = filter_obj.gaussian_notch_reject(image, notch_positions, radius)

            assert ideal.shape == shape
            assert butterworth.shape == shape
            assert gaussian.shape == shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
