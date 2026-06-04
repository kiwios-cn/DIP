"""
图像算术和逻辑操作测试用例
测试所有算术和逻辑操作的正确性
"""

import pytest
import numpy as np
import cv2
from image_arithmetic_logic import ImageArithmeticLogic


class TestImageArithmeticLogic:
    """测试图像算术和逻辑操作类"""

    @pytest.fixture
    def processor(self):
        """创建处理器实例"""
        return ImageArithmeticLogic()

    @pytest.fixture
    def test_images(self):
        """创建测试图像"""
        # 创建简单的测试图像
        image1 = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        image2 = np.array([[50, 100], [150, 200]], dtype=np.uint8)
        return image1, image2

    @pytest.fixture
    def test_images_color(self):
        """创建彩色测试图像"""
        image1 = np.ones((2, 2, 3), dtype=np.uint8) * 100
        image2 = np.ones((2, 2, 3), dtype=np.uint8) * 50
        return image1, image2

    # ==================== 算术操作测试 ====================

    def test_add_basic(self, processor, test_images):
        """测试基本加法操作"""
        image1, image2 = test_images
        result = processor.add(image1, image2, 1.0, 1.0)

        # 验证结果形状
        assert result.shape == image1.shape

        # 验证加法结果（注意溢出处理）
        expected = np.array([[150, 250], [255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_add_weighted(self, processor, test_images):
        """测试加权加法"""
        image1, image2 = test_images
        result = processor.add(image1, image2, 0.5, 0.5)

        # 验证加权平均
        expected = np.array([[75, 125], [175, 225]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_subtract_basic(self, processor, test_images):
        """测试基本减法操作"""
        image1, image2 = test_images
        result = processor.subtract(image1, image2)

        # 验证减法结果（注意下溢处理）
        expected = np.array([[50, 50], [50, 50]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_subtract_underflow(self, processor):
        """测试减法下溢保护"""
        image1 = np.array([[50, 100]], dtype=np.uint8)
        image2 = np.array([[100, 50]], dtype=np.uint8)
        result = processor.subtract(image1, image2)

        # 验证下溢被截断到0
        expected = np.array([[0, 50]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_multiply_basic(self, processor):
        """测试基本乘法操作"""
        image1 = np.array([[100, 200]], dtype=np.uint8)
        image2 = np.array([[2, 1]], dtype=np.uint8)
        result = processor.multiply(image1, image2, 1.0/255.0)

        # 验证乘法结果
        # 100*2/255 ≈ 0.78 → 0, 200*1/255 ≈ 0.78 → 0
        assert result.shape == image1.shape
        assert result.dtype == np.uint8

    def test_multiply_overflow(self, processor):
        """测试乘法溢出保护"""
        image1 = np.array([[200, 255]], dtype=np.uint8)
        image2 = np.array([[200, 255]], dtype=np.uint8)
        result = processor.multiply(image1, image2, 1.0)

        # 验证溢出被截断到255
        expected = np.array([[255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_divide_basic(self, processor):
        """测试基本除法操作"""
        image1 = np.array([[100, 200]], dtype=np.uint8)
        image2 = np.array([[2, 4]], dtype=np.uint8)
        result = processor.divide(image1, image2, 1.0)

        # 验证除法结果
        expected = np.array([[50, 50]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_divide_by_zero(self, processor):
        """测试除零保护"""
        image1 = np.array([[100, 200]], dtype=np.uint8)
        image2 = np.array([[0, 0]], dtype=np.uint8)
        result = processor.divide(image1, image2, 1.0)

        # 验证除零不会导致错误，结果应该是原图像
        expected = np.array([[100, 200]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    # ==================== 逻辑操作测试 ====================

    def test_bitwise_and(self, processor):
        """测试按位与操作"""
        # 使用二进制易于验证的值
        image1 = np.array([[0b11110000, 0b10101010]], dtype=np.uint8)
        image2 = np.array([[0b11001100, 0b11001100]], dtype=np.uint8)
        result = processor.bitwise_and(image1, image2)

        expected = np.array([[0b11000000, 0b10001000]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_bitwise_or(self, processor):
        """测试按位或操作"""
        image1 = np.array([[0b11110000, 0b10101010]], dtype=np.uint8)
        image2 = np.array([[0b11001100, 0b11001100]], dtype=np.uint8)
        result = processor.bitwise_or(image1, image2)

        expected = np.array([[0b11111100, 0b11101110]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_bitwise_not(self, processor):
        """测试按位非操作"""
        image = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        result = processor.bitwise_not(image)

        expected = np.array([[255, 0], [127, 191]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_bitwise_xor(self, processor):
        """测试按位异或操作"""
        image1 = np.array([[0b11110000, 0b10101010]], dtype=np.uint8)
        image2 = np.array([[0b11001100, 0b11001100]], dtype=np.uint8)
        result = processor.bitwise_xor(image1, image2)

        expected = np.array([[0b00111100, 0b01100110]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    # ==================== 彩色图像测试 ====================

    def test_add_color_images(self, processor, test_images_color):
        """测试彩色图像加法"""
        image1, image2 = test_images_color
        result = processor.add(image1, image2, 0.5, 0.5)

        assert result.shape == image1.shape
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_logic_color_images(self, processor, test_images_color):
        """测试彩色图像逻辑操作"""
        image1, image2 = test_images_color

        and_result = processor.bitwise_and(image1, image2)
        or_result = processor.bitwise_or(image1, image2)
        xor_result = processor.bitwise_xor(image1, image2)

        assert and_result.shape == image1.shape
        assert or_result.shape == image1.shape
        assert xor_result.shape == image1.shape

    # ==================== 边界情况测试 ====================

    def test_all_zeros(self, processor):
        """测试全零图像"""
        image1 = np.zeros((10, 10), dtype=np.uint8)
        image2 = np.zeros((10, 10), dtype=np.uint8)

        add_result = processor.add(image1, image2)
        and_result = processor.bitwise_and(image1, image2)

        assert np.all(add_result == 0)
        assert np.all(and_result == 0)

    def test_all_ones(self, processor):
        """测试全255图像"""
        image1 = np.ones((10, 10), dtype=np.uint8) * 255
        image2 = np.ones((10, 10), dtype=np.uint8) * 255

        or_result = processor.bitwise_or(image1, image2)
        not_result = processor.bitwise_not(image1)

        assert np.all(or_result == 255)
        assert np.all(not_result == 0)

    def test_same_image_operations(self, processor, test_images):
        """测试相同图像的操作"""
        image1, _ = test_images

        # 相同图像异或应该为0
        xor_result = processor.bitwise_xor(image1, image1)
        assert np.all(xor_result == 0)

        # 相同图像与操作应该等于原图像
        and_result = processor.bitwise_and(image1, image1)
        np.testing.assert_array_equal(and_result, image1)

    # ==================== 加载图像测试 ====================

    def test_load_single_image(self, processor, tmp_path):
        """测试加载单张图像"""
        # 创建临时测试图像
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), test_img)

        # 测试加载
        success = processor.load_images(str(img_path))
        assert success
        assert processor.image1 is not None

    def test_load_two_images(self, processor, tmp_path):
        """测试加载两张图像"""
        # 创建临时测试图像
        test_img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        test_img2 = np.ones((100, 100, 3), dtype=np.uint8) * 64
        img_path1 = tmp_path / "test1.png"
        img_path2 = tmp_path / "test2.png"
        cv2.imwrite(str(img_path1), test_img1)
        cv2.imwrite(str(img_path2), test_img2)

        # 测试加载
        success = processor.load_images(str(img_path1), str(img_path2))
        assert success
        assert processor.image1 is not None
        assert processor.image2 is not None

    def test_load_nonexistent_image(self, processor):
        """测试加载不存在的图像"""
        success = processor.load_images("nonexistent.png")
        assert not success

    def test_load_different_size_images(self, processor, tmp_path):
        """测试加载不同尺寸的图像（应自动调整）"""
        test_img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        test_img2 = np.ones((200, 200, 3), dtype=np.uint8) * 64
        img_path1 = tmp_path / "test1.png"
        img_path2 = tmp_path / "test2.png"
        cv2.imwrite(str(img_path1), test_img1)
        cv2.imwrite(str(img_path2), test_img2)

        success = processor.load_images(str(img_path1), str(img_path2))
        assert success
        # 验证第二张图像被调整为与第一张相同尺寸
        assert processor.image1.shape == processor.image2.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
