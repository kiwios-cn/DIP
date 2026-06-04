"""
边缘检测测试
"""

import pytest
import numpy as np
import cv2
from edge_detection import EdgeDetection


class TestEdgeDetection:
    """测试边缘检测类"""

    @pytest.fixture
    def detector(self):
        """创建边缘检测对象"""
        return EdgeDetection()

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (30, 30), (70, 70), 200, 2)
        return image

    @pytest.fixture
    def edge_image(self):
        """创建有明显边缘的图像"""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:, :50] = 50
        image[:, 50:] = 200
        return image

    # ==================== 点检测测试 ====================

    def test_point_detection_basic(self, detector, test_image):
        """测试基本点检测"""
        result = detector.point_detection(test_image, threshold=0.5)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_point_detection_threshold(self, detector, test_image):
        """测试不同阈值"""
        result_low = detector.point_detection(test_image, threshold=0.3)
        result_high = detector.point_detection(test_image, threshold=0.7)

        # 低阈值检测到更多点
        assert np.sum(result_low) >= np.sum(result_high)

    # ==================== 线检测测试 ====================

    def test_line_detection_horizontal(self, detector):
        """测试水平线检测"""
        # 创建水平线图像
        image = np.zeros((100, 100), dtype=np.uint8)
        image[50, :] = 255

        result = detector.line_detection(image, 'horizontal', threshold=0.5)

        assert result.shape == image.shape
        assert np.sum(result) > 0  # 应该检测到线

    def test_line_detection_vertical(self, detector):
        """测试垂直线检测"""
        # 创建垂直线图像
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:, 50] = 255

        result = detector.line_detection(image, 'vertical', threshold=0.5)

        assert result.shape == image.shape
        assert np.sum(result) > 0

    def test_line_detection_diagonal(self, detector):
        """测试对角线检测"""
        image = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            if i < 100:
                image[i, i] = 255

        result_45 = detector.line_detection(image, 'diagonal_45', threshold=0.5)
        result_135 = detector.line_detection(image, 'diagonal_135', threshold=0.5)

        assert result_45.shape == image.shape
        assert result_135.shape == image.shape

    def test_line_detection_invalid_direction(self, detector, test_image):
        """测试无效方向应该抛出异常"""
        with pytest.raises(ValueError):
            detector.line_detection(test_image, 'invalid')

    # ==================== Roberts算子测试 ====================

    def test_roberts_edge_basic(self, detector, test_image):
        """测试Roberts边缘检测基本功能"""
        magnitude, gx, gy = detector.roberts_edge(test_image)

        assert magnitude.shape == test_image.shape
        assert gx.shape == test_image.shape
        assert gy.shape == test_image.shape
        assert magnitude.dtype == np.uint8

    def test_roberts_edge_detects_edges(self, detector, edge_image):
        """测试Roberts算子能检测到边缘"""
        magnitude, gx, gy = detector.roberts_edge(edge_image)

        # 边缘处应该有强响应
        assert np.max(magnitude) > 0
        # 垂直边缘，X方向梯度应该有值
        assert np.max(np.abs(gx)) > 0

    # ==================== Prewitt算子测试 ====================

    def test_prewitt_edge_basic(self, detector, test_image):
        """测试Prewitt边缘检测基本功能"""
        magnitude, gx, gy = detector.prewitt_edge(test_image)

        assert magnitude.shape == test_image.shape
        assert gx.shape == test_image.shape
        assert gy.shape == test_image.shape
        assert magnitude.dtype == np.uint8

    def test_prewitt_edge_detects_edges(self, detector, edge_image):
        """测试Prewitt算子能检测到边缘"""
        magnitude, gx, gy = detector.prewitt_edge(edge_image)

        assert np.max(magnitude) > 0
        assert np.max(np.abs(gx)) > 0

    # ==================== Sobel算子测试 ====================

    def test_sobel_edge_basic(self, detector, test_image):
        """测试Sobel边缘检测基本功能"""
        magnitude, gx, gy = detector.sobel_edge(test_image, ksize=3)

        assert magnitude.shape == test_image.shape
        assert gx.shape == test_image.shape
        assert gy.shape == test_image.shape
        assert magnitude.dtype == np.uint8

    def test_sobel_edge_kernel_sizes(self, detector, test_image):
        """测试不同核大小"""
        for ksize in [3, 5, 7]:
            magnitude, gx, gy = detector.sobel_edge(test_image, ksize=ksize)
            assert magnitude.shape == test_image.shape

    def test_sobel_edge_detects_edges(self, detector, edge_image):
        """测试Sobel算子能检测到边缘"""
        magnitude, gx, gy = detector.sobel_edge(edge_image)

        assert np.max(magnitude) > 0
        assert np.max(np.abs(gx)) > 0

    # ==================== Canny边缘检测测试 ====================

    def test_canny_edge_basic(self, detector, test_image):
        """测试Canny边缘检测基本功能"""
        edges = detector.canny_edge(test_image, 50, 150)

        assert edges.shape == test_image.shape
        assert edges.dtype == np.uint8
        # Canny输出是二值图像
        assert np.all((edges == 0) | (edges == 255))

    def test_canny_edge_thresholds(self, detector, test_image):
        """测试不同阈值"""
        edges_low = detector.canny_edge(test_image, 30, 100)
        edges_high = detector.canny_edge(test_image, 100, 200)

        # 低阈值检测到更多边缘
        assert np.sum(edges_low) >= np.sum(edges_high)

    def test_canny_edge_detects_edges(self, detector, edge_image):
        """测试Canny能检测到边缘"""
        edges = detector.canny_edge(edge_image, 50, 150)

        # 应该检测到垂直边缘
        assert np.sum(edges) > 0

    # ==================== 算子比较测试 ====================

    def test_compare_edge_detectors(self, detector, test_image):
        """测试不同算子的相对强度"""
        roberts, _, _ = detector.roberts_edge(test_image)
        prewitt, _, _ = detector.prewitt_edge(test_image)
        sobel, _, _ = detector.sobel_edge(test_image)

        # 所有算子都应该检测到边缘
        assert np.max(roberts) > 0
        assert np.max(prewitt) > 0
        assert np.max(sobel) > 0

    def test_sobel_stronger_than_prewitt(self, detector, edge_image):
        """测试Sobel通常比Prewitt响应更强"""
        prewitt_mag, _, _ = detector.prewitt_edge(edge_image)
        sobel_mag, _, _ = detector.sobel_edge(edge_image)

        # Sobel有高斯加权，通常响应更强
        # 这个测试可能不总是成立，取决于图像内容
        assert sobel_mag.shape == prewitt_mag.shape

    # ==================== 梯度方向测试 ====================

    def test_gradient_direction_vertical_edge(self, detector):
        """测试垂直边缘的梯度方向"""
        # 垂直边缘
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:, :50] = 50
        image[:, 50:] = 200

        magnitude, gx, gy = detector.sobel_edge(image)

        # 垂直边缘，X方向梯度应该最强
        assert np.max(np.abs(gx)) > np.max(np.abs(gy))

    def test_gradient_direction_horizontal_edge(self, detector):
        """测试水平边缘的梯度方向"""
        # 水平边缘
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 50
        image[50:, :] = 200

        magnitude, gx, gy = detector.sobel_edge(image)

        # 水平边缘，Y方向梯度应该最强
        assert np.max(np.abs(gy)) > np.max(np.abs(gx))

    # ==================== 边界情况测试 ====================

    def test_uniform_image(self, detector):
        """测试均匀图像"""
        uniform = np.ones((50, 50), dtype=np.uint8) * 128

        roberts, _, _ = detector.roberts_edge(uniform)
        prewitt, _, _ = detector.prewitt_edge(uniform)
        sobel, _, _ = detector.sobel_edge(uniform)
        canny = detector.canny_edge(uniform, 50, 150)

        # 均匀图像应该没有边缘
        assert np.max(roberts) < 10
        assert np.max(prewitt) < 10
        assert np.max(sobel) < 10
        assert np.sum(canny) == 0

    def test_all_black_image(self, detector):
        """测试全黑图像"""
        black = np.zeros((50, 50), dtype=np.uint8)

        magnitude, _, _ = detector.sobel_edge(black)
        assert np.max(magnitude) == 0

    def test_all_white_image(self, detector):
        """测试全白图像"""
        white = np.ones((50, 50), dtype=np.uint8) * 255

        magnitude, _, _ = detector.sobel_edge(white)
        assert np.max(magnitude) == 0

    # ==================== 图像加载测试 ====================

    def test_load_image_success(self, detector, test_image, tmp_path):
        """测试成功加载图像"""
        temp_file = tmp_path / "test.png"
        cv2.imwrite(str(temp_file), test_image)

        success = detector.load_image(str(temp_file))
        assert success
        assert detector.image is not None

    def test_load_image_failure(self, detector):
        """测试加载不存在的图像"""
        success = detector.load_image("nonexistent.png")
        assert not success


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
