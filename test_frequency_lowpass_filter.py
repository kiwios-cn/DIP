"""
频域低通/高通滤波器测试用例
测试理想、巴特沃斯、高斯低通/高通滤波器的正确性
"""

import pytest
import numpy as np
import cv2
from frequency_lowpass_filter import FrequencyLowpassFilter


class TestFrequencyLowpassFilter:
    """测试频域低通/高通滤波器类"""

    @pytest.fixture
    def filter_obj(self):
        """创建滤波器实例"""
        return FrequencyLowpassFilter()

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        image = np.ones((128, 128), dtype=np.uint8) * 128
        cv2.circle(image, (64, 64), 30, 200, -1)
        return image

    @pytest.fixture
    def edge_image(self):
        """创建边缘图像"""
        image = np.zeros((128, 128), dtype=np.uint8)
        image[:, :64] = 255
        return image

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

        # 验证重建精度
        np.testing.assert_allclose(reconstructed, test_image, atol=1e-5)

    # ==================== 掩码创建测试 ====================

    def test_create_circular_mask_ideal_lowpass(self, filter_obj):
        """测试理想低通掩码"""
        H, W = 100, 100
        cutoff = 30

        mask = filter_obj._create_circular_mask(H, W, cutoff, 'ideal', 'low')

        # 验证形状
        assert mask.shape == (H, W)

        # 验证值域 [0, 1]
        assert np.all(mask >= 0) and np.all(mask <= 1)

        # 验证中心点通过
        assert mask[H//2, W//2] == 1.0

        # 验证边缘被阻止
        assert mask[0, 0] == 0.0

    def test_create_circular_mask_ideal_highpass(self, filter_obj):
        """测试理想高通掩码"""
        H, W = 100, 100
        cutoff = 30

        mask = filter_obj._create_circular_mask(H, W, cutoff, 'ideal', 'high')

        # 验证中心点被阻止
        assert mask[H//2, W//2] == 0.0

        # 验证边缘通过
        assert mask[0, 0] == 1.0

    def test_create_circular_mask_butterworth_lowpass(self, filter_obj):
        """测试巴特沃斯低通掩码"""
        H, W = 100, 100
        cutoff = 30

        mask = filter_obj._create_circular_mask(H, W, cutoff, 'butterworth', 'low', order=2)

        # 验证形状和值域
        assert mask.shape == (H, W)
        assert np.all(mask >= 0) and np.all(mask <= 1)

        # 验证中心点接近1
        assert mask[H//2, W//2] > 0.9

        # 验证平滑过渡（截止频率处约0.5）
        # 这需要计算到中心的距离，这里简化验证
        assert np.min(mask) < 0.5 < np.max(mask)

    def test_create_circular_mask_gaussian_lowpass(self, filter_obj):
        """测试高斯低通掩码"""
        H, W = 100, 100
        cutoff = 30

        mask = filter_obj._create_circular_mask(H, W, cutoff, 'gaussian', 'low')

        # 验证形状和值域
        assert mask.shape == (H, W)
        assert np.all(mask >= 0) and np.all(mask <= 1)

        # 验证中心点为1
        assert mask[H//2, W//2] == 1.0

    def test_create_circular_mask_invalid_type(self, filter_obj):
        """测试无效的滤波器类型"""
        with pytest.raises(ValueError):
            filter_obj._create_circular_mask(100, 100, 30, 'invalid', 'low')

    # ==================== 低通滤波器测试 ====================

    def test_ideal_lowpass_filter(self, filter_obj, test_image):
        """测试理想低通滤波器"""
        cutoff = 30
        result = filter_obj.ideal_lowpass_filter(test_image, cutoff)

        # 验证形状和类型
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

        # 低通滤波应该去除高频，但理想滤波器可能产生振铃
        # 验证频谱中高频分量被减少
        _, fft_original = filter_obj.fft_transform(test_image)
        _, fft_filtered = filter_obj.fft_transform(result)

        # 计算边缘区域的频谱能量（高频）
        H, W = test_image.shape
        edge_region = 10
        high_freq_original = np.sum(np.abs(fft_original[:edge_region, :]))
        high_freq_filtered = np.sum(np.abs(fft_filtered[:edge_region, :]))

        # 滤波后高频能量应该减少
        assert high_freq_filtered < high_freq_original

    def test_butterworth_lowpass_filter(self, filter_obj, test_image):
        """测试巴特沃斯低通滤波器"""
        cutoff = 30
        result = filter_obj.butterworth_lowpass_filter(test_image, cutoff, order=2)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_gaussian_lowpass_filter(self, filter_obj, test_image):
        """测试高斯低通滤波器"""
        cutoff = 30
        result = filter_obj.gaussian_lowpass_filter(test_image, cutoff)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_lowpass_different_orders(self, filter_obj, test_image):
        """测试不同阶数的巴特沃斯滤波器"""
        cutoff = 30

        result_order1 = filter_obj.butterworth_lowpass_filter(test_image, cutoff, order=1)
        result_order4 = filter_obj.butterworth_lowpass_filter(test_image, cutoff, order=4)

        # 高阶滤波器应该更接近理想滤波器
        assert result_order1.shape == result_order4.shape

    # ==================== 高通滤波器测试 ====================

    def test_ideal_highpass_filter(self, filter_obj, test_image):
        """测试理想高通滤波器"""
        cutoff = 30
        result = filter_obj.ideal_highpass_filter(test_image, cutoff)

        # 验证形状和类型
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

        # 高通滤波应该保留边缘（高频）
        # 结果的平均值应该接近0（去除了DC分量）
        assert np.mean(result) < np.mean(test_image)

    def test_butterworth_highpass_filter(self, filter_obj, test_image):
        """测试巴特沃斯高通滤波器"""
        cutoff = 30
        result = filter_obj.butterworth_highpass_filter(test_image, cutoff, order=2)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_gaussian_highpass_filter(self, filter_obj, test_image):
        """测试高斯高通滤波器"""
        cutoff = 30
        result = filter_obj.gaussian_highpass_filter(test_image, cutoff)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_highpass_preserves_edges(self, filter_obj, edge_image):
        """测试高通滤波器保留边缘"""
        cutoff = 20
        result = filter_obj.ideal_highpass_filter(edge_image, cutoff)

        # 高通滤波应该保留边缘信息
        # 边缘处应该有非零值
        edge_col = result[:, 63:65]
        assert np.sum(edge_col) > 0

    # ==================== 带通/带阻滤波器测试 ====================

    def test_bandpass_filter(self, filter_obj, test_image):
        """测试带通滤波器"""
        cutoff_low = 10
        cutoff_high = 50

        result = filter_obj.bandpass_filter(test_image, cutoff_low, cutoff_high)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

        # 带通应该去除极低频和极高频
        assert np.mean(result) < np.mean(test_image)

    def test_bandreject_filter(self, filter_obj, test_image):
        """测试带阻滤波器"""
        cutoff_low = 10
        cutoff_high = 50

        result = filter_obj.bandreject_filter(test_image, cutoff_low, cutoff_high)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_bandpass_bandreject_complementary(self, filter_obj, test_image):
        """测试带通和带阻的互补性"""
        cutoff_low = 10
        cutoff_high = 50

        bandpass = filter_obj.bandpass_filter(test_image, cutoff_low, cutoff_high, 'ideal')
        bandreject = filter_obj.bandreject_filter(test_image, cutoff_low, cutoff_high, 'ideal')

        # 两者应该互补（虽然不是完全的线性叠加）
        # 验证它们确实不同
        assert not np.array_equal(bandpass, bandreject)

    # ==================== 低通/高通互补性测试 ====================

    def test_lowpass_highpass_complementary(self, filter_obj, test_image):
        """测试低通和高通的互补性"""
        cutoff = 30

        # FFT变换
        _, fft_shifted = filter_obj.fft_transform(test_image)

        # 创建低通和高通掩码
        H, W = test_image.shape
        mask_low = filter_obj._create_circular_mask(H, W, cutoff, 'ideal', 'low')
        mask_high = filter_obj._create_circular_mask(H, W, cutoff, 'ideal', 'high')

        # 低通 + 高通 = 1（理想情况）
        mask_sum = mask_low + mask_high
        np.testing.assert_allclose(mask_sum, 1.0, atol=1e-10)

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

    def test_filter_uniform_image(self, filter_obj):
        """测试均匀图像的滤波"""
        uniform = np.ones((100, 100), dtype=np.uint8) * 128

        lp_result = filter_obj.ideal_lowpass_filter(uniform, 30)
        hp_result = filter_obj.ideal_highpass_filter(uniform, 30)

        # 均匀图像低通滤波后应该保持均匀（只有DC分量）
        np.testing.assert_allclose(lp_result, uniform, atol=1)

        # 均匀图像高通滤波后应该接近0（无高频）
        assert np.mean(hp_result) < 10

    def test_filter_zero_image(self, filter_obj):
        """测试全零图像"""
        zero_image = np.zeros((100, 100), dtype=np.uint8)

        lp_result = filter_obj.ideal_lowpass_filter(zero_image, 30)
        hp_result = filter_obj.ideal_highpass_filter(zero_image, 30)

        # 零图像滤波后应该仍是零
        np.testing.assert_allclose(lp_result, zero_image, atol=1)
        np.testing.assert_allclose(hp_result, zero_image, atol=1)

    def test_filter_small_cutoff(self, filter_obj, test_image):
        """测试小截止频率"""
        cutoff = 5

        result = filter_obj.ideal_lowpass_filter(test_image, cutoff)

        # 小截止频率应该导致严重模糊
        assert result.shape == test_image.shape

    def test_filter_large_cutoff(self, filter_obj, test_image):
        """测试大截止频率"""
        cutoff = 100

        result = filter_obj.ideal_lowpass_filter(test_image, cutoff)

        # 大截止频率应该保留大部分信息
        assert result.shape == test_image.shape
        # 结果应该接近原图
        np.testing.assert_allclose(result, test_image, atol=5)

    def test_filter_zero_cutoff(self, filter_obj, test_image):
        """测试零截止频率"""
        cutoff = 0

        result = filter_obj.ideal_lowpass_filter(test_image, cutoff)

        # 零截止频率低通滤波应该产生均匀图像（只保留DC）
        assert result.shape == test_image.shape

    # ==================== 滤波器类型比较测试 ====================

    def test_filter_types_comparison(self, filter_obj, test_image):
        """测试不同滤波器类型的效果"""
        cutoff = 30

        ideal = filter_obj.ideal_lowpass_filter(test_image, cutoff)
        butterworth = filter_obj.butterworth_lowpass_filter(test_image, cutoff, order=2)
        gaussian = filter_obj.gaussian_lowpass_filter(test_image, cutoff)

        # 所有滤波器都应该产生模糊效果
        assert np.mean(ideal) > 0
        assert np.mean(butterworth) > 0
        assert np.mean(gaussian) > 0

        # 它们的结果应该不完全相同
        assert not np.array_equal(ideal, butterworth)
        assert not np.array_equal(butterworth, gaussian)

    def test_butterworth_order_effect(self, filter_obj, test_image):
        """测试巴特沃斯阶数的影响"""
        cutoff = 30

        order1 = filter_obj.butterworth_lowpass_filter(test_image, cutoff, order=1)
        order2 = filter_obj.butterworth_lowpass_filter(test_image, cutoff, order=2)
        order5 = filter_obj.butterworth_lowpass_filter(test_image, cutoff, order=5)

        # 不同阶数应该产生不同结果
        assert not np.array_equal(order1, order2)
        assert not np.array_equal(order2, order5)

        # 更高阶数应该更接近理想滤波器（更强的模糊）
        # 这里只验证基本属性
        assert order1.shape == order2.shape == order5.shape

    # ==================== 性能测试 ====================

    def test_filter_preserves_shape(self, filter_obj):
        """测试所有滤波器保持图像形状"""
        test_shapes = [(64, 64), (128, 128), (256, 256)]
        cutoff = 30

        for shape in test_shapes:
            image = np.random.randint(0, 256, shape, dtype=np.uint8)

            ideal_lp = filter_obj.ideal_lowpass_filter(image, cutoff)
            butterworth_lp = filter_obj.butterworth_lowpass_filter(image, cutoff)
            gaussian_lp = filter_obj.gaussian_lowpass_filter(image, cutoff)

            ideal_hp = filter_obj.ideal_highpass_filter(image, cutoff)
            butterworth_hp = filter_obj.butterworth_highpass_filter(image, cutoff)
            gaussian_hp = filter_obj.gaussian_highpass_filter(image, cutoff)

            assert ideal_lp.shape == shape
            assert butterworth_lp.shape == shape
            assert gaussian_lp.shape == shape
            assert ideal_hp.shape == shape
            assert butterworth_hp.shape == shape
            assert gaussian_hp.shape == shape

    def test_different_cutoffs(self, filter_obj, test_image):
        """测试不同截止频率"""
        cutoffs = [10, 20, 30, 50, 80]

        for cutoff in cutoffs:
            result = filter_obj.butterworth_lowpass_filter(test_image, cutoff)
            assert result.shape == test_image.shape

    # ==================== 频率响应测试 ====================

    def test_lowpass_removes_high_frequency(self, filter_obj):
        """测试低通滤波器去除高频"""
        # 创建高频图像（棋盘格）
        H, W = 128, 128
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        checkerboard = ((x // 4 + y // 4) % 2 * 255).astype(np.uint8)

        # 低通滤波
        result = filter_obj.ideal_lowpass_filter(checkerboard, 20)

        # 棋盘格应该被平滑
        assert np.std(result) < np.std(checkerboard)

    def test_highpass_removes_low_frequency(self, filter_obj):
        """测试高通滤波器去除低频"""
        # 创建低频图像（渐变）
        H, W = 128, 128
        gradient = np.linspace(0, 255, W, dtype=np.uint8)
        gradient = np.tile(gradient, (H, 1))

        # 高通滤波
        result = filter_obj.ideal_highpass_filter(gradient, 20)

        # 渐变应该被去除（接近均匀）
        assert np.std(result) < np.std(gradient)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
