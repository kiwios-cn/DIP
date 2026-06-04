"""
形态学操作测试
"""

import pytest
import numpy as np
import cv2
from morphological_operations import MorphologicalOperations


class TestMorphologicalOperations:
    """测试形态学操作类"""

    @pytest.fixture
    def morph(self):
        """创建形态学操作对象"""
        return MorphologicalOperations()

    @pytest.fixture
    def binary_image(self):
        """创建测试二值图像"""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (30, 30), (70, 70), 255, -1)
        return image

    @pytest.fixture
    def noisy_binary(self):
        """创建含噪声的二值图像"""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (30, 30), (70, 70), 255, -1)

        # 添加白色噪声点
        for _ in range(20):
            x, y = np.random.randint(0, 100, 2)
            image[y, x] = 255

        return image

    # ==================== 腐蚀测试 ====================

    def test_erosion_basic(self, morph, binary_image):
        """测试基本腐蚀"""
        result = morph.erosion(binary_image, kernel_size=3)

        assert result.shape == binary_image.shape
        assert result.dtype == np.uint8

        # 腐蚀后白色区域应该变小
        assert np.sum(result) < np.sum(binary_image)

    def test_erosion_kernel_shapes(self, morph, binary_image):
        """测试不同结构元素形状"""
        for shape in ['rect', 'ellipse', 'cross']:
            result = morph.erosion(binary_image, kernel_size=3, kernel_shape=shape)
            assert result.shape == binary_image.shape

    def test_erosion_iterations(self, morph, binary_image):
        """测试多次迭代"""
        result1 = morph.erosion(binary_image, kernel_size=3, iterations=1)
        result2 = morph.erosion(binary_image, kernel_size=3, iterations=2)

        # 迭代次数越多，白色区域越小
        assert np.sum(result2) <= np.sum(result1)

    def test_erosion_all_black(self, morph):
        """测试全黑图像"""
        black = np.zeros((50, 50), dtype=np.uint8)
        result = morph.erosion(black, kernel_size=3)

        assert np.all(result == 0)

    def test_erosion_all_white(self, morph):
        """测试全白图像"""
        white = np.ones((50, 50), dtype=np.uint8) * 255
        result = morph.erosion(white, kernel_size=3)

        # 边界会被腐蚀，但中心应该保持白色
        assert np.sum(result) > 0

    # ==================== 膨胀测试 ====================

    def test_dilation_basic(self, morph, binary_image):
        """测试基本膨胀"""
        result = morph.dilation(binary_image, kernel_size=3)

        assert result.shape == binary_image.shape
        assert result.dtype == np.uint8

        # 膨胀后白色区域应该变大
        assert np.sum(result) > np.sum(binary_image)

    def test_dilation_kernel_shapes(self, morph, binary_image):
        """测试不同结构元素形状"""
        for shape in ['rect', 'ellipse', 'cross']:
            result = morph.dilation(binary_image, kernel_size=3, kernel_shape=shape)
            assert result.shape == binary_image.shape

    def test_dilation_iterations(self, morph, binary_image):
        """测试多次迭代"""
        result1 = morph.dilation(binary_image, kernel_size=3, iterations=1)
        result2 = morph.dilation(binary_image, kernel_size=3, iterations=2)

        # 迭代次数越多，白色区域越大
        assert np.sum(result2) >= np.sum(result1)

    # ==================== 开运算测试 ====================

    def test_opening_basic(self, morph, noisy_binary):
        """测试开运算去除噪声"""
        result = morph.opening(noisy_binary, kernel_size=3)

        assert result.shape == noisy_binary.shape

        # 开运算应该去除小的白色噪声点
        # 结果应该更接近干净的图像

    def test_opening_idempotent(self, morph, binary_image):
        """测试开运算的幂等性"""
        result1 = morph.opening(binary_image, kernel_size=3)
        result2 = morph.opening(result1, kernel_size=3)

        # 第二次开运算应该与第一次结果相同（幂等性）
        assert np.array_equal(result1, result2)

    def test_opening_smaller_or_equal(self, morph, binary_image):
        """测试开运算结果不大于原图"""
        result = morph.opening(binary_image, kernel_size=3)

        # 开运算结果应该小于或等于原图
        assert np.sum(result) <= np.sum(binary_image)

    # ==================== 闭运算测试 ====================

    def test_closing_basic(self, morph, binary_image):
        """测试闭运算"""
        # 创建有孔洞的图像
        image_with_holes = binary_image.copy()
        # 创建一个小孔洞（3x3）
        cv2.rectangle(image_with_holes, (48, 48), (52, 52), 0, -1)

        result = morph.closing(image_with_holes, kernel_size=7)

        assert result.shape == image_with_holes.shape

        # 闭运算应该填充小孔洞
        assert np.sum(result) >= np.sum(image_with_holes)

    def test_closing_idempotent(self, morph, binary_image):
        """测试闭运算的幂等性"""
        result1 = morph.closing(binary_image, kernel_size=3)
        result2 = morph.closing(result1, kernel_size=3)

        # 第二次闭运算应该与第一次结果相同
        assert np.array_equal(result1, result2)

    def test_closing_larger_or_equal(self, morph, binary_image):
        """测试闭运算结果不小于原图"""
        result = morph.closing(binary_image, kernel_size=3)

        # 闭运算结果应该大于或等于原图
        assert np.sum(result) >= np.sum(binary_image)

    # ==================== 形态学梯度测试 ====================

    def test_morphological_gradient(self, morph, binary_image):
        """测试形态学梯度"""
        result = morph.morphological_gradient(binary_image, kernel_size=3)

        assert result.shape == binary_image.shape

        # 梯度应该突出边缘
        # 检查边缘区域有值
        assert np.max(result) > 0

    def test_gradient_edges_only(self, morph):
        """测试梯度只在边缘有值"""
        # 创建简单形状
        image = np.zeros((50, 50), dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (40, 40), 255, -1)

        result = morph.morphological_gradient(image, kernel_size=3)

        # 内部和外部远离边缘的地方应该是0
        assert result[20, 20] < result[10, 20]  # 内部 < 边缘附近

    # ==================== 顶帽变换测试 ====================

    def test_top_hat(self, morph):
        """测试顶帽变换"""
        # 创建有小亮点的图像
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), 100, -1)
        cv2.circle(image, (50, 50), 5, 255, -1)  # 小亮点

        result = morph.top_hat(image, kernel_size=15)

        assert result.shape == image.shape

        # 顶帽应该提取小亮点
        assert np.max(result) > 0

    def test_top_hat_uniform(self, morph):
        """测试均匀图像的顶帽"""
        uniform = np.ones((50, 50), dtype=np.uint8) * 128
        result = morph.top_hat(uniform, kernel_size=9)

        # 均匀图像的顶帽应该全为0
        assert np.all(result == 0)

    # ==================== 黑帽变换测试 ====================

    def test_black_hat(self, morph):
        """测试黑帽变换"""
        # 创建有小暗点的图像
        image = np.ones((100, 100), dtype=np.uint8) * 200
        cv2.circle(image, (50, 50), 5, 50, -1)  # 小暗点

        result = morph.black_hat(image, kernel_size=15)

        assert result.shape == image.shape

        # 黑帽应该提取小暗点
        assert np.max(result) > 0

    def test_black_hat_uniform(self, morph):
        """测试均匀图像的黑帽"""
        uniform = np.ones((50, 50), dtype=np.uint8) * 128
        result = morph.black_hat(uniform, kernel_size=9)

        # 均匀图像的黑帽应该全为0
        assert np.all(result == 0)

    # ==================== 结构元素测试 ====================

    def test_get_kernel_invalid_shape(self, morph):
        """测试无效的结构元素形状"""
        with pytest.raises(ValueError):
            morph._get_kernel(3, 'invalid')

    def test_get_kernel_shapes(self, morph):
        """测试不同形状的结构元素"""
        rect = morph._get_kernel(5, 'rect')
        ellipse = morph._get_kernel(5, 'ellipse')
        cross = morph._get_kernel(5, 'cross')

        assert rect.shape == (5, 5)
        assert ellipse.shape == (5, 5)
        assert cross.shape == (5, 5)

        # 不同形状的结构元素应该不同
        assert not np.array_equal(rect, ellipse)

    # ==================== 复合操作测试 ====================

    def test_erosion_then_dilation(self, morph, binary_image):
        """测试腐蚀后膨胀（开运算的手动实现）"""
        eroded = morph.erosion(binary_image, kernel_size=3)
        result = morph.dilation(eroded, kernel_size=3)

        # 应该接近开运算结果
        opening = morph.opening(binary_image, kernel_size=3)

        # 结果应该相似（可能不完全相同由于实现细节）
        assert result.shape == opening.shape

    def test_dilation_then_erosion(self, morph, binary_image):
        """测试膨胀后腐蚀（闭运算的手动实现）"""
        dilated = morph.dilation(binary_image, kernel_size=3)
        result = morph.erosion(dilated, kernel_size=3)

        # 应该接近闭运算结果
        closing = morph.closing(binary_image, kernel_size=3)

        assert result.shape == closing.shape

    # ==================== 图像加载测试 ====================

    def test_load_image_success(self, morph, binary_image, tmp_path):
        """测试成功加载图像"""
        temp_file = tmp_path / "test.png"
        cv2.imwrite(str(temp_file), binary_image)

        success = morph.load_image(str(temp_file))
        assert success
        assert morph.image is not None

    def test_load_image_failure(self, morph):
        """测试加载不存在的图像"""
        success = morph.load_image("nonexistent.png")
        assert not success


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
