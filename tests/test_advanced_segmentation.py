"""
高级图像分割测试
"""

import pytest
import numpy as np
import cv2
from advanced_segmentation import AdvancedSegmentation


class TestAdvancedSegmentation:
    """测试高级图像分割类"""

    @pytest.fixture
    def seg(self):
        """创建分割对象"""
        return AdvancedSegmentation()

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (30, 30), (70, 70), 200, 2)
        return image

    @pytest.fixture
    def edge_image(self):
        """创建边缘图像"""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), 255, 1)
        return image

    # ==================== Canny边缘检测测试 ====================

    def test_canny_edge_detection(self, seg, test_image):
        """测试Canny边缘检测"""
        edges, intermediate = seg.canny_edge_detection(test_image, 50, 150)

        assert edges.shape == test_image.shape
        assert edges.dtype == np.uint8
        assert 'blurred' in intermediate
        assert 'magnitude' in intermediate
        assert 'nms' in intermediate

    def test_canny_detects_edges(self, seg, test_image):
        """测试Canny能检测到边缘"""
        edges, _ = seg.canny_edge_detection(test_image, 50, 150)
        assert np.sum(edges) > 0

    # ==================== Hough变换测试 ====================

    def test_hough_lines(self, seg, edge_image):
        """测试Hough直线检测"""
        lines = seg.hough_lines(edge_image, threshold=50)
        assert isinstance(lines, list)

    def test_hough_circles(self, seg):
        """测试Hough圆检测"""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (50, 50), 30, 200, 2)

        circles = seg.hough_circles(image, min_radius=20, max_radius=40)
        assert isinstance(circles, list)

    # ==================== OTSU阈值测试 ====================

    def test_otsu_threshold(self, seg, test_image):
        """测试OTSU阈值"""
        threshold, binary = seg.otsu_threshold(test_image)

        assert isinstance(threshold, int)
        assert 0 <= threshold <= 255
        assert binary.shape == test_image.shape
        assert np.all((binary == 0) | (binary == 255))

    def test_otsu_bimodal_image(self, seg):
        """测试双峰图像的OTSU"""
        # 创建双峰图像
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 50
        image[50:, :] = 200

        threshold, binary = seg.otsu_threshold(image)
        # 阈值应该在两个峰之间或等于边界值
        assert 50 <= threshold <= 200

    # ==================== 分水岭分割测试 ====================

    def test_watershed_segmentation(self, seg):
        """测试分水岭分割"""
        # 创建彩色测试图像
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(image, (30, 30), 20, (255, 255, 255), -1)
        cv2.circle(image, (70, 70), 20, (255, 255, 255), -1)

        result, markers = seg.watershed_segmentation(image, auto_markers=True)

        assert result.shape == image.shape
        assert markers is not None

    # ==================== 辅助方法测试 ====================

    def test_load_image_success(self, seg, test_image, tmp_path):
        """测试成功加载图像"""
        temp_file = tmp_path / "test.png"
        cv2.imwrite(str(temp_file), test_image)

        success = seg.load_image(str(temp_file))
        assert success
        assert seg.image is not None

    def test_load_image_failure(self, seg):
        """测试加载不存在的图像"""
        success = seg.load_image("nonexistent.png")
        assert not success


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
