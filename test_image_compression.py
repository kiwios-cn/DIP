"""
图像压缩编码测试
"""

import pytest
import numpy as np
import cv2
from image_compression import (
    HuffmanCoding, ArithmeticCoding, LZWCoding, BitPlaneCoding,
    CompressionComparison
)


class TestHuffmanCoding:
    """测试哈夫曼编码"""

    @pytest.fixture
    def huffman(self):
        return HuffmanCoding()

    @pytest.fixture
    def test_image(self):
        return np.array([[100, 100, 150, 150],
                        [100, 100, 150, 150],
                        [200, 200, 250, 250],
                        [200, 200, 250, 250]], dtype=np.uint8)

    def test_encode_decode(self, huffman, test_image):
        """测试编码和解码"""
        encoded, metadata = huffman.encode(test_image)
        decoded = huffman.decode(encoded, metadata)

        assert np.array_equal(test_image, decoded)

    def test_frequency_table(self, huffman, test_image):
        """测试频率表"""
        freq_table = huffman.build_frequency_table(test_image)

        assert freq_table[100] == 4
        assert freq_table[150] == 4
        assert freq_table[200] == 4
        assert freq_table[250] == 4

    def test_compression_ratio(self, huffman, test_image):
        """测试压缩比计算"""
        encoded, metadata = huffman.encode(test_image)
        ratio = huffman.calculate_compression_ratio(test_image, encoded)

        assert ratio > 0

    def test_single_value(self, huffman):
        """测试单一值图像"""
        image = np.ones((4, 4), dtype=np.uint8) * 128
        encoded, metadata = huffman.encode(image)
        decoded = huffman.decode(encoded, metadata)

        assert np.array_equal(image, decoded)


class TestArithmeticCoding:
    """测试算术编码"""

    @pytest.fixture
    def arithmetic(self):
        return ArithmeticCoding()

    @pytest.fixture
    def test_image(self):
        return np.array([[100, 150], [200, 250]], dtype=np.uint8)

    def test_probability_table(self, arithmetic, test_image):
        """测试概率表"""
        prob_table = arithmetic.build_probability_table(test_image)

        assert len(prob_table) == 4
        assert sum(prob_table.values()) == pytest.approx(1.0)

    def test_cumulative_table(self, arithmetic, test_image):
        """测试累积概率表"""
        prob_table = arithmetic.build_probability_table(test_image)
        cumulative = arithmetic.build_cumulative_table(prob_table)

        # 检查累积概率范围
        for low, high in cumulative.values():
            assert 0 <= low < high <= 1

    def test_encode(self, arithmetic, test_image):
        """测试编码"""
        code, metadata = arithmetic.encode(test_image)

        assert isinstance(code, float)
        assert 0 <= code <= 1


class TestLZWCoding:
    """测试LZW编码"""

    @pytest.fixture
    def lzw(self):
        return LZWCoding()

    @pytest.fixture
    def test_image(self):
        return np.array([[100, 100, 150, 150],
                        [100, 100, 150, 150]], dtype=np.uint8)

    def test_encode_decode(self, lzw, test_image):
        """测试编码和解码"""
        encoded, metadata = lzw.encode(test_image)
        decoded = lzw.decode(encoded, metadata)

        assert np.array_equal(test_image, decoded)

    def test_compression_ratio(self, lzw, test_image):
        """测试压缩比"""
        encoded, metadata = lzw.encode(test_image)
        ratio = lzw.calculate_compression_ratio(test_image, encoded)

        assert ratio > 0

    def test_repeated_pattern(self, lzw):
        """测试重复模式"""
        # 创建有重复模式的图像
        pattern = np.array([1, 2, 3, 4], dtype=np.uint8)
        image = np.tile(pattern, (4, 4))

        encoded, metadata = lzw.encode(image)
        decoded = lzw.decode(encoded, metadata)

        assert np.array_equal(image, decoded)


class TestBitPlaneCoding:
    """测试位平面编码"""

    @pytest.fixture
    def bitplane(self):
        return BitPlaneCoding()

    @pytest.fixture
    def test_image(self):
        return np.array([[0, 255, 0, 255],
                        [255, 0, 255, 0]], dtype=np.uint8)

    def test_decompose_bit_planes(self, bitplane, test_image):
        """测试位平面分解"""
        bit_planes = bitplane.decompose_bit_planes(test_image)

        assert len(bit_planes) == 8

        # 检查每个位平面只包含0和1
        for plane in bit_planes:
            assert np.all((plane == 0) | (plane == 1))

    def test_run_length_encode_decode(self, bitplane):
        """测试游程编码和解码"""
        data = np.array([1, 1, 1, 0, 0, 1, 1, 0], dtype=np.uint8)
        runs = bitplane.run_length_encode(data)
        decoded = bitplane.run_length_decode(runs, len(data))

        assert np.array_equal(data, decoded)

    def test_encode_decode(self, bitplane, test_image):
        """测试完整编码和解码"""
        encoded, metadata = bitplane.encode(test_image)
        decoded = bitplane.decode(encoded, metadata)

        assert np.array_equal(test_image, decoded)

    def test_color_image_error(self, bitplane):
        """测试彩色图像应该抛出异常"""
        color_image = np.zeros((4, 4, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            bitplane.encode(color_image)

    def test_uniform_image(self, bitplane):
        """测试均匀图像"""
        image = np.ones((8, 8), dtype=np.uint8) * 128

        encoded, metadata = bitplane.encode(image)
        decoded = bitplane.decode(encoded, metadata)

        assert np.array_equal(image, decoded)


class TestCompressionComparison:
    """测试压缩算法比较"""

    @pytest.fixture
    def comparison(self):
        return CompressionComparison()

    @pytest.fixture
    def test_image(self):
        return np.random.randint(0, 256, (16, 16), dtype=np.uint8)

    def test_compare_all(self, comparison, test_image):
        """测试所有算法比较"""
        results = comparison.compare_all(test_image)

        # 检查所有算法都有结果
        assert 'Huffman' in results
        assert 'LZW' in results
        assert 'BitPlane' in results

        # 检查成功的算法有压缩比
        for method, result in results.items():
            if result['status'] == 'success' and 'ratio' in result:
                assert result['ratio'] > 0


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_image(self):
        """测试空图像"""
        empty = np.array([], dtype=np.uint8).reshape(0, 0)

        # 哈夫曼编码应该能处理空图像
        huffman = HuffmanCoding()
        with pytest.raises(Exception):
            huffman.encode(empty)

    def test_single_pixel(self):
        """测试单像素图像"""
        single = np.array([[128]], dtype=np.uint8)

        huffman = HuffmanCoding()
        encoded, metadata = huffman.encode(single)
        decoded = huffman.decode(encoded, metadata)

        assert np.array_equal(single, decoded)

    def test_large_image(self):
        """测试大图像"""
        large = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        huffman = HuffmanCoding()
        encoded, metadata = huffman.encode(large)
        decoded = huffman.decode(encoded, metadata)

        assert np.array_equal(large, decoded)

    def test_binary_image(self):
        """测试二值图像"""
        binary = np.random.randint(0, 2, (32, 32), dtype=np.uint8) * 255

        lzw = LZWCoding()
        encoded, metadata = lzw.encode(binary)
        decoded = lzw.decode(encoded, metadata)

        assert np.array_equal(binary, decoded)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
